#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <getopt.h>
#include <iomanip>  
#include <filesystem>  
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <csignal>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// 전역 변수 
cv::Mat sharedFrame;
std::vector<cv::Rect> sharedDetections;
std::vector<std::vector<float>> saved_embeddings;
std::atomic<bool> running(true);
std::atomic<bool> door_unlocked(false);
std::mutex frameMutex, detectionMutex, embeddingMutex;
std::string face_detection_model_path = "../assets/model/yolov8n_face.onnx";
std::string face_classification_model_path = "../assets/model/recognition_resnet27.onnx";
std::string verification_status = " "; 
std::string owner=" ";
std::string embedding_name = "unk";  // 기본값
int port = 9090;
int cam_id = 0;

inline std::vector<float> preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb;
    resize(frame, resized, Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cvtColor(resized, rgb, COLOR_BGR2RGB);

    std::vector<float> tensor(3 * 320 * 320);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 320; ++y)
            for (int x = 0; x < 320; ++x)
                tensor[idx++] = rgb.at<Vec3f>(y, x)[c];
    return tensor;
}


inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < 512; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.f;
}

void saveEmbeddingsToFile(const std::vector<std::vector<float>>& embeddings, const std::string& embedding_name) {
    std::string save_path = "../assets/face_information/" + embedding_name + ".raw";
    std::ofstream ofs(save_path, std::ios::binary);
    for (const auto& emb : embeddings) {
        ofs.write(reinterpret_cast<const char*>(emb.data()), 512 * sizeof(float));
    }
    ofs.close();
    std::cout << "[INFO] Saved " << embeddings.size() << " embeddings to " << save_path << "\n";
}



void cameraThread(int client_sock) {
    VideoCapture cap(cam_id);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Camera open failed!" << endl;
        running = false;
        return;
    }

    Mat frame;
    std::vector<cv::Rect> local_detections;
    bool first_shown = false;
    auto first_show_time = std::chrono::steady_clock::now();

    while (running) {
        cap >> frame;
        if (frame.empty()) continue;

        resize(frame, frame, Size(320, 320));

        auto now = std::chrono::steady_clock::now();
        if (!first_shown) {
            first_show_time = now;
            first_shown = true;
        }

        double elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - first_show_time).count();

        if (elapsed < 10.0) {
            int remaining = 10 - static_cast<int>(elapsed);
            putText(frame, "Please look at the screen.", Point(10, 250), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
            putText(frame, "Registration starts in " + std::to_string(remaining) + " seconds.", Point(10, 275), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
            putText(frame, "Make sure no one else is visible.", Point(10, 300), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
        } else {
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                sharedFrame = frame.clone();
            }

            if (detectionMutex.try_lock()) {
                local_detections = sharedDetections;
                detectionMutex.unlock();
            }

            if (local_detections.size() > 1) {
                putText(frame, "Too many people detected.", Point(10, 275), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1);
                putText(frame, "Please move to a private area.", Point(10, 300), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 0, 255), 1);
            }

            for (const auto& rect : local_detections)
                rectangle(frame, rect, Scalar(255, 255, 255), 2);

            putText(frame, verification_status, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);

            {
                std::lock_guard<std::mutex> lock(embeddingMutex);
                int num_embeds = static_cast<int>(saved_embeddings.size());
                int percent = std::min(100, (num_embeds * 100) / 100);  // 100개 기준

                std::string rate_text = "Progress: " + std::to_string(percent) + "% ";
                putText(frame, rate_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                if (num_embeds >= 100) {
                    putText(frame, "Enough data is collected", Point(10, 275), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
                    putText(frame, "You can close the manager", Point(10, 300), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(255, 255, 255), 1);
                }
            }
        }

        // 프레임 전송
        std::vector<uchar> buf;
        if (cv::imencode(".jpg", frame, buf)) {
            uint32_t len = buf.size();
            uint32_t len_net = htonl(len);

            ssize_t sent_len = send(client_sock, &len_net, sizeof(len_net), MSG_NOSIGNAL);
            if (sent_len != sizeof(len_net)) {
                std::cerr << "[INFO] Failed to send frame size: " << strerror(errno) << std::endl;
                break; 
            }

            ssize_t sent_data = send(client_sock, buf.data(), len, MSG_NOSIGNAL);
            if (sent_data != static_cast<ssize_t>(len)) {
                std::cerr << "[INFO] Failed to send full frame: " << strerror(errno) << std::endl;
                break; 
            }
        } else {
            std::cerr << "[INFO] Failed to encode frame." << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cap.release();
    shutdown(client_sock, SHUT_RDWR);
    close(client_sock);

    {
        std::lock_guard<std::mutex> lock(embeddingMutex);
        if (!saved_embeddings.empty()) {
            saveEmbeddingsToFile(saved_embeddings, embedding_name);
        }
    }

}


void faceDetectionThread(Ort::Session& session) {
    std::vector<std::string> input_name_strs = session.GetInputNames();
    std::vector<std::string> output_name_strs = session.GetOutputNames();

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    for (const auto& name : input_name_strs) input_names.push_back(name.c_str());
    for (const auto& name : output_name_strs) output_names.push_back(name.c_str());

    std::array<int64_t, 4> input_shape{1, 3, 320, 320};

    while (running) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (sharedFrame.empty()) continue;
            frame = sharedFrame.clone();
        }

        std::vector<float> input_tensor = preprocess(frame);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value ort_input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor.data(),
            input_tensor.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &ort_input_tensor, 1,
                                          output_names.data(), 1);

        const float* output = output_tensors[0].GetTensorMutableData<float>();

        const int num_preds = 2100;
        const float conf_thres = 0.5f, nms_thres = 0.45f;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;

        for (int i = 0; i < num_preds; ++i) {
            float x = output[0 * num_preds + i];
            float y = output[1 * num_preds + i];
            float w = output[2 * num_preds + i];
            float h = output[3 * num_preds + i];
            float score = output[4 * num_preds + i];

            if (score > conf_thres) {
                int x1 = static_cast<int>(x - w / 2);
                int y1 = static_cast<int>(y - h / 2);
                int x2 = static_cast<int>(x + w / 2);
                int y2 = static_cast<int>(y + h / 2);
                if (x1 > 0 && y1 > 0 && x2 < frame.cols && y2 < frame.rows) {
                    boxes.emplace_back(cv::Point(x1, y1), cv::Point(x2, y2));
                    scores.push_back(score);
                }
            }
        }

        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, scores, conf_thres, nms_thres, indices);
        }

        std::vector<cv::Rect> result;
        for (int i : indices) result.push_back(boxes[i]);

        {
            std::lock_guard<std::mutex> lock(detectionMutex);
            sharedDetections = std::move(result);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


void saveEmbeddingsThread(Ort::Session& session, const std::string& save_name) {
    std::vector<std::string> input_name_strs = session.GetInputNames();
    std::vector<std::string> output_name_strs = session.GetOutputNames();
    const char* input_name = input_name_strs[0].c_str();
    const char* output_name = output_name_strs[0].c_str();
    std::array<int64_t, 4> shape{1, 3, 128, 128};

    while (running) {
        cv::Mat frame;
        std::vector<cv::Rect> faces;
        {
            std::lock_guard<std::mutex> lock1(frameMutex);
            std::lock_guard<std::mutex> lock2(detectionMutex);
            if (sharedFrame.empty() || sharedDetections.empty()) continue;
            frame = sharedFrame.clone();
            faces = sharedDetections;
        }

        for (const auto& rect : faces) {
            cv::Rect r = rect & cv::Rect(0, 0, frame.cols, frame.rows);
            if (r.width < 10 || r.height < 10) continue;

            cv::Mat face = frame(r).clone();
            resize(face, face, Size(128, 128));
            cvtColor(face, face, COLOR_BGR2GRAY);
            std::vector<cv::Mat> chs(3, face);
            cv::Mat rgb;
            merge(chs, rgb);
            rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

            std::vector<float> input_tensor(3 * 128 * 128);
            size_t idx = 0;
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < 128; ++y)
                    for (int x = 0; x < 128; ++x)
                        input_tensor[idx++] = rgb.at<cv::Vec3f>(y, x)[c];

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto tensor_val = Ort::Value::CreateTensor<float>(mem_info, input_tensor.data(), input_tensor.size(), shape.data(), shape.size());

            auto outputs = session.Run(Ort::RunOptions{nullptr}, &input_name, &tensor_val, 1, &output_name, 1);
            const float* emb_out = outputs[0].GetTensorMutableData<float>();

            std::vector<float> curr_emb(emb_out, emb_out + 512);

            {
                std::lock_guard<std::mutex> lock(embeddingMutex);
                saved_embeddings.push_back(curr_emb);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void signalHandler(int signum) {
    running = false;
}

void parseArgs(int argc, char* argv[]) {
    static struct option long_opts[] = {
        {"face_detection_model_path", required_argument, nullptr, 'f'},
        {"face_classification_model_path", required_argument, nullptr, 'm'},
        {"cam", required_argument, nullptr, 'c'},
        {"port", required_argument, nullptr, 'p'},
        {"name", required_argument, nullptr, 'n'},  // 🔹 추가됨
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:m:c:p:n:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'f': face_detection_model_path = optarg; break;
            case 'm': face_classification_model_path = optarg; break;
            case 'c': cam_id = atoi(optarg); break;
            case 'p': port = atoi(optarg); break;
            case 'n': embedding_name = optarg; break;  // 🔹 추가됨
            case 'h':
            default:
                cout << "Usage: " << argv[0]
                     << " --face_detection_model_path <path>\n"
                     << " --face_classification_model_path <path>\n"
                     << " --cam <id>\n"
                     << " --port <port>\n"
                     << " --name <embedding_name>\n";
                exit(0);
        }
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    parseArgs(argc, argv);

    // ONNX Runtime 환경 초기화
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "face-register-app");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session det_sess(env, face_detection_model_path.c_str(), options);
    Ort::Session cls_sess(env, face_classification_model_path.c_str(), options);

    // ✅ 클라이언트 수신 소켓 초기화
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[ERROR] Failed to create socket\n";
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[ERROR] Bind failed\n";
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "[ERROR] Listen failed\n";
        close(server_fd);
        return 1;
    }

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_sock = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_sock < 0) {
        std::cerr << "[ERROR] Accept failed\n";
        close(server_fd);
        return 1;
    }

    // ✅ 쓰레드 실행
    std::thread camThread(cameraThread, client_sock);
    std::thread detThread(faceDetectionThread, std::ref(det_sess));
    std::thread clsThread(saveEmbeddingsThread, std::ref(cls_sess), embedding_name);

    // 쓰레드 종료 대기
    camThread.join();
    detThread.join();
    clsThread.join();

    close(server_fd);
    std::cout << "[INFO] All threads finished. Program exiting.\n";
    return 0;
}


// int main(int argc, char* argv[]) {
//     signal(SIGINT, signalHandler);
//     signal(SIGTERM, signalHandler);

//     parseArgs(argc, argv);

//     Ort::Env env(ORT_LOGGING_LEVEL_INFOING, "face-register-app");
//     Ort::SessionOptions options;
//     options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

//     Ort::Session det_sess(env, face_detection_model_path.c_str(), options);
//     Ort::Session cls_sess(env, face_classification_model_path.c_str(), options);

//     std::thread camThread(cameraThread);
//     std::thread detThread(faceDetectionThread, std::ref(det_sess));
//     std::thread clsThread(saveEmbeddingsThread, std::ref(cls_sess), embedding_name);  // ✅ 수정됨

//     camThread.join();
//     detThread.join();
//     clsThread.join();

//     return 0;
// }

