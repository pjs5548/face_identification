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
#include <termios.h>
#include <signal.h>
#include <cstdlib> 
#include <sstream>
#include <omp.h>  
#include <json.hpp>  


// 전역 변수 
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std;
using namespace cv;

cv::Mat sharedFrame;
std::vector<cv::Rect> sharedDetections;
std::vector<std::vector<float>> saved_embeddings;
std::atomic<bool> running(true);
std::atomic<bool> door_unlocked(false);
std::atomic<bool> camera_running(false);
std::mutex frameMutex, detectionMutex, embeddingMutex;
std::string face_detection_model_path = "../assets/model/yolov8n_face.onnx";
std::string face_classification_model_path = "../assets/model/recognition_resnet27.onnx";
std::chrono::steady_clock::time_point sharedFrameTime;

std::string verification_status = " "; 
std::string owner=" ";
std::atomic<bool> verification_status_changed(false); 
int PORT = 9070;
int cam_id = 0;
static struct termios oldt;


inline std::vector<float> preprocess(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(320, 320));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB); // inplace 가능
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);  // channels[0]=R, [1]=G, [2]=B

    std::vector<float> tensor(3 * 320 * 320);

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 320; ++y) {
            const float* row_ptr = channels[c].ptr<float>(y);
            for (int x = 0; x < 320; ++x) {
                tensor[c * 320 * 320 + y * 320 + x] = row_ptr[x];
            }
        }
    }

    return tensor;
}

inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.f, na = 0.f, nb = 0.f;

    #pragma omp parallel for reduction(+:dot,na,nb)
    for (int i = 0; i < 512; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    return (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.f;
}

std::vector<std::vector<float>> loadEmbeddings(const std::string& path) {
    std::vector<std::vector<float>> result;
    std::ifstream ifs(path, ios::binary);
    if (!ifs.is_open()) return result;

    while (ifs.peek() != EOF) {
        std::vector<float> emb(512);
        ifs.read(reinterpret_cast<char*>(emb.data()), 512 * sizeof(float));
        if (ifs.gcount() == 512 * sizeof(float)) result.emplace_back(std::move(emb));
    }
    return result;
}


void cameraThread() {
    VideoCapture cap;

    while (running) {
        if (door_unlocked.load() && !camera_running.load()) {
            cap.open(cam_id);
            if (!cap.isOpened()) {
                cerr << "[ERROR] Camera open failed!" << endl;
                running = false;
                return;
            }
            camera_running.store(true);
        } else if (!door_unlocked.load() && camera_running.load()) {
            cap.release();
            camera_running.store(false);
            sharedFrame.release();
        }

        if (camera_running.load()) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) continue;

            resize(frame, frame, Size(320, 320));
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                sharedFrame = frame.clone();
                sharedFrameTime = std::chrono::steady_clock::now();  // 🔥 타임스탬프 저장
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (cap.isOpened()) cap.release();
}

void socketSenderThread() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        running = false;
        return;
    }

    sockaddr_in address{};
    socklen_t addrlen = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        running = false;
        return;
    }

    if (listen(server_fd, 1) < 0) {
        perror("listen failed");
        close(server_fd);
        running = false;
        return;
    }

    int client_sock = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if (client_sock < 0) {
        perror("accept failed");
        close(server_fd);
        running = false;
        return;
    }

    while (running) {
        if (!door_unlocked.load()) {
            if (verification_status != " ") {
                verification_status = " ";
                verification_status_changed = true;
                std::cout << "Door locked: verification_status cleared.\n";
            }
        }

        cv::Mat frame;
        std::chrono::steady_clock::time_point frame_time;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (sharedFrame.empty()) continue;
            frame = sharedFrame.clone();
            frame_time = sharedFrameTime;
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - frame_time).count() > 10) {
            continue;  // 오래된 프레임 무시
        }

        Scalar boxColor = (verification_status == "NOT VERIFIED") ? Scalar(255, 255, 255)
                            : (verification_status == " ")       ? Scalar(255, 255, 255)
                                                                 : Scalar(255, 255, 255);

        for (const auto& rect : sharedDetections)
            rectangle(frame, rect, boxColor, 2);

        putText(frame, verification_status, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, boxColor, 2);

        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer);

        int32_t size = buffer.size();
        int32_t net_size = htonl(size);

        // 헤더: JSON 유무 표시 (1 byte)
        uint8_t has_json = verification_status_changed ? 1 : 0;
        if (send(client_sock, &has_json, 1, 0) != 1) break;

        // 이미지 전송
        if (send(client_sock, &net_size, sizeof(net_size), 0) != sizeof(net_size)) break;
        if (send(client_sock, buffer.data(), size, 0) != size) break;

        // 인증 결과 전송
        if (has_json) {
            json j;
            j["user"] = verification_status;
            j["verified"] = (verification_status != " " && verification_status != "NOT VERIFIED");
            j["door"] = door_unlocked.load() ? "unlocked" : "locked";

            std::string json_str = j.dump();
            int32_t json_len = htonl(json_str.size());

            if (send(client_sock, &json_len, sizeof(json_len), 0) != sizeof(json_len)) break;
            if (send(client_sock, json_str.data(), json_str.size(), 0) != (ssize_t)json_str.size()) break;

            verification_status_changed = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    close(client_sock);
    close(server_fd);
}


void faceDetectionThread(Ort::Session& session) {
    std::vector<std::string> input_name_strs = session.GetInputNames();
    std::vector<std::string> output_name_strs = session.GetOutputNames();

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    for (const auto& name : input_name_strs) input_names.push_back(name.c_str());
    for (const auto& name : output_name_strs) output_names.push_back(name.c_str());

    std::array<int64_t, 4> input_shape{1, 3, 320, 320};

    int frame_cnt=0;



    while (running && door_unlocked.load()) {
        cv::Mat frame;
        if (++frame_cnt % 3 != 0) continue;
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
        if (!indices.empty()) {
            int max_area = 0;
            int best_idx = -1;
            for (int i : indices) {
                int area = boxes[i].area();
                if (area > max_area) {
                    max_area = area;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) {
                result.push_back(boxes[best_idx]);
            }
        }

        {
            std::lock_guard<std::mutex> lock(detectionMutex);
            sharedDetections = std::move(result);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void faceClassificationThread(Ort::Session& session, std::chrono::steady_clock::time_point start_time) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::vector<std::vector<float>>>> owners;

    for (const auto& entry : fs::directory_iterator("../assets/face_information/")) {
        if (entry.is_regular_file() && entry.path().extension() == ".raw") {
            std::string filename = entry.path().stem().string();
            auto embeddings = loadEmbeddings(entry.path().string());
            if (!embeddings.empty()) {
                owners.emplace_back(filename, embeddings);
            }
        }
    }

    if (owners.empty()) {
        std::cerr << "[INFO] No registered faces found!\n";
        return;
    }

    std::vector<std::string> input_name_strs = session.GetInputNames();
    std::vector<std::string> output_name_strs = session.GetOutputNames();
    const char* input_name = input_name_strs[0].c_str();
    const char* output_name = output_name_strs[0].c_str();
    std::array<int64_t, 4> shape{1, 3, 128, 128};

    float best_sim = 0.0f;
    std::string best_owner;
    int frame_count = 0;

    auto perform_verification = [&](const std::vector<std::vector<float>>& refs,
                                    const std::string& owner_name) -> void {
        if (!door_unlocked.load()) return;  // 문 잠기면 바로 중단

        int total = 0, pass = 0;
        float local_best_sim = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        const int max_duration_ms = 5000;
        while (running && door_unlocked.load() && total < 5) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count() > max_duration_ms) {
                break;
            }

            cv::Mat frame;
            std::vector<cv::Rect> faces;
            {
                std::lock_guard<std::mutex> lock1(frameMutex);
                std::lock_guard<std::mutex> lock2(detectionMutex);
                if (sharedFrame.empty() || sharedDetections.empty()) continue;
                frame = sharedFrame.clone();
                faces = sharedDetections;
            }

            if (++frame_count % 3 != 0) continue;

            for (const auto& rect : faces) {
                if (!door_unlocked.load()) return;  //검사 중에도 잠기면 중단
                if (total >= 5) return;

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
                    saved_embeddings.emplace_back(curr_emb);
                }

                float max_sim = 0.f;
                for (const auto& ref : refs)
                    max_sim = std::max(max_sim, cosineSimilarity(curr_emb, ref));

                ++total;
                if (max_sim > 0.5f) {
                    ++pass;
                } 

                if (max_sim > local_best_sim) {
                    local_best_sim = max_sim;
                }

                if (pass >= 5 && local_best_sim > best_sim) {
                    best_sim = local_best_sim;
                    best_owner = owner_name;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    if (!door_unlocked.load()) return;  // 문이 잠겨있으면 시작도 안 함

    for (auto& [owner, embeddings] : owners) {
        if (!door_unlocked.load()) break;
        perform_verification(embeddings, owner);
    }

    if (!door_unlocked.load()) return;  // retry 전에 또 확인

    if (best_owner.empty()) {
        std::cout << "[INFO] First verification failed. Retrying after 5 seconds...\n";
        verification_status_changed = true;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        for (auto& [owner, embeddings] : owners) {
            if (!door_unlocked.load()) break;
            perform_verification(embeddings, owner);
        }
    }

    if (!door_unlocked.load()) return;

    if (!best_owner.empty()) {
        verification_status = best_owner;
        verification_status_changed = true;
        system("aplay ../assets/sound/success.wav");
    } else {
        verification_status_changed = true;
        system("aplay ../assets/sound/fail.wav");
    }
}




// 시그널 핸들러
void restoreTerminalOnSignal(int signo) {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    exit(0);
}

void buttonThread() {
    struct termios newt;

    if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
        perror("tcgetattr failed");
        return;
    }

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);

    if (tcsetattr(STDIN_FILENO, TCSANOW, &newt) != 0) {
        perror("tcsetattr failed");
        return;
    }

    signal(SIGINT, restoreTerminalOnSignal);
    signal(SIGTERM, restoreTerminalOnSignal);

    std::cout << "[INFO] Press 'u' to unlock, 'l' to lock\n";

    while (running) {
        char c = 0;
        ssize_t bytes_read = read(STDIN_FILENO, &c, 1);
        if (bytes_read > 0) {
            if (c == 'u' || c == 'U') {
                if (!door_unlocked.load()) {
                    door_unlocked.store(true);
                }
            } else if (c == 'l' || c == 'L') {
                if (door_unlocked.load()) {
                    door_unlocked.store(false);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (tcsetattr(STDIN_FILENO, TCSANOW, &oldt) != 0) {
        perror("tcsetattr restore failed");
    }
}

void parseArgs(int argc, char* argv[]) {
    static struct option long_opts[] = {
        {"face_detection_model_path", required_argument, nullptr, 'f'},
        {"face_classification_model_path", required_argument, nullptr, 'm'},
        {"cam", required_argument, nullptr, 'c'},
        {"port", required_argument, nullptr, 'p'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "f:m:c:p:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'f': face_detection_model_path = optarg; break;
            case 'm': face_classification_model_path = optarg; break;
            case 'c': cam_id = atoi(optarg); break;
            case 'p': PORT = atoi(optarg); break;
            case 'h':
            default:
                std::cout << "Usage: --face_detection_model_path <path> "
                             "--face_classification_model_path <path> "
                             "--cam <id> --port <port>\n";
                exit(0);
        }
    }
}

int main(int argc, char* argv[]) {
    parseArgs(argc, argv);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "face-id-app");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    options.SetInterOpNumThreads(std::thread::hardware_concurrency());
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session det_sess(env, face_detection_model_path.c_str(), options);
    Ort::Session cls_sess(env, face_classification_model_path.c_str(), options);

    std::thread camThread(cameraThread);
    std::thread sockThread(socketSenderThread);
    std::thread btnThread(buttonThread);

    std::thread detThread, clsThread;
    bool detectionStarted = false;

    while (running) {
        bool unlocked = door_unlocked.load();

        if (unlocked && !detectionStarted) {
            std::cout << "[INFO] Door unlocked! Starting face identification logic\n";

            auto start_time = std::chrono::steady_clock::now();
            detThread = std::thread(faceDetectionThread, std::ref(det_sess));
            clsThread = std::thread(faceClassificationThread, std::ref(cls_sess), start_time);

            detectionStarted = true;

        } else if (!unlocked && detectionStarted) {
            std::cout << "[INFO] Door locked! Stopping face identification logic\n";

            if (detThread.joinable()) detThread.join();
            if (clsThread.joinable()) clsThread.join();

            detectionStarted = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    camThread.join();
    sockThread.join();
    btnThread.join();
    if (detThread.joinable()) detThread.join();
    if (clsThread.joinable()) clsThread.join();

    return 0;
}
