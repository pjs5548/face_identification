import socket
import struct
import cv2
import numpy as np
import json
import argparse
import subprocess
import time
import signal

def recv_all(sock, length):
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9070, help='Server port')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    args = parser.parse_args()

    # C++ 서버 실행
    try:
        server_process = subprocess.Popen([
            "../bin/face_identification",
            "--port", str(args.port),
            "--cam", str(args.cam)
        ])
    except Exception as e:
        print(f"[ERROR] C++ server execution fail: {e}")
        return

    time.sleep(1)  # 서버 준비 대기

    # 클라이언트 소켓 연결
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", args.port))
    except Exception as e:
        print(f"[ERROR] Server connection fail: {e}")
        server_process.terminate()
        return

    window_name = "Viewer"
    door_status = "unlocked"
    window_visible = True
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 항상 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키
                break
            if window_visible and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 프레임 수신
            has_json_buf = recv_all(sock, 1)
            if not has_json_buf:
                break
            has_json = bool(struct.unpack('!B', has_json_buf)[0])

            length_buf = recv_all(sock, 4)
            if not length_buf:
                break
            img_len = struct.unpack('!I', length_buf)[0]

            img_data = recv_all(sock, img_len)
            if not img_data:
                break

            frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # JSON 수신 및 상태 업데이트
            if has_json:
                json_len_buf = recv_all(sock, 4)
                if not json_len_buf:
                    break
                json_len = struct.unpack('!I', json_len_buf)[0]

                json_data = recv_all(sock, json_len)
                if not json_data:
                    break

                try:
                    info = json.loads(json_data.decode('utf-8'))
                    print(f"[INFO] User: {info['user']} / Verified: {info['verified']}")
                    new_door_status = info.get("door", door_status)
                    if new_door_status != door_status:
                        door_status = new_door_status
                        if door_status == "locked" and window_visible:
                            cv2.destroyWindow(window_name)
                            window_visible = False
                        elif door_status == "unlocked" and not window_visible:
                            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                            window_visible = True
                except Exception as e:
                    print(f"[INFO] JSON parsing fail: {e}")

            # 이미지 표시
            if door_status == "unlocked" and window_visible:
                cv2.imshow(window_name, frame)

    finally:
        if window_visible:
            cv2.destroyAllWindows()
        sock.close()
        if server_process.poll() is None:
            server_process.send_signal(signal.SIGINT)
            time.sleep(1)
            server_process.terminate()

if __name__ == '__main__':
    main()
