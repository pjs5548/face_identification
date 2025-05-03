# Face Identification System

## 1. 필수 패키지 설치

Ubuntu 기반 시스템에서 아래 명령어를 통해 필수 패키지를 설치하세요:

sudo apt update
sudo apt install -y \\
    build-essential cmake pkg-config \\
    libjpeg-dev libpng-dev libtiff-dev \\
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \\
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \\
    zlib1g-dev libprotobuf-dev protobuf-compiler \\
    nlohmann-json-dev libomp-dev libpthread-stubs0-dev \\
    alsa-utils \\
    python3 python3-pip python3-pyqt5 libgl1

Python 패키지 설치:

pip install numpy==1.26.4 opencv-python==4.11.0 pyqt5==5.15.11

---
## 2. 기본 설정
1. 주어진 링크에서 recognition_resnet27.onnx 파일을 다운 받으세요
2. deepinsight/assets/model에 recognition_resnet27.onnx 를 다운 받아서 ㅓ흐세요

---
## 3. 얼굴 등록 절차

1. deepinsight/python 폴더로 이동 후 다음 명령어 실행:

    python3 gui.py --port xxxx

    - xxxx는 9000~9999 사이의 포트 번호 (예: 9100)

2. "Face Registration Manager" GUI 창이 열립니다.

3. 중앙에 있는 리스트는 등록된 얼굴의 임베딩 .raw 파일 목록입니다.

4. 등록된 파일을 클릭 후 Remove를 누르면 해당 파일이 삭제됩니다.

5. Remove All 버튼을 클릭하면 모든 .raw 파일이 삭제됩니다.

6. 얼굴을 등록하려면 Registeration 버튼 클릭 → 이니셜 입력 → 확인 → 10초 대기 후 바운딩 박스가 표시됩니다.

7. 등록 완료 후 GUI 창 우측 상단 X를 클릭하여 종료합니다.

8. 등록된 얼굴 임베딩은 assets/face_information 폴더에 이니셜.raw 파일로 저장됩니다.

---

## 4. 얼굴 인증 절차

1. deepinsight/bin 폴더로 이동하여 다음 명령어 실행: ./face_identifiaction --cam 0 --port xxxx
    - --cam: 카메라 ID (예: 0)
    - --port: 9000~9999 사이의 포트 번호

2. u 키를 누르면 도어락이 해제되며 얼굴 인식이 시작됩니다.

3. 1차 인식 성공 시 즉시 패스, 실패 시 5초 후 재시도합니다.

4. 인식 성공 시 터미널에 사용자 이니셜이 표시되고 imshow 창에도 표시됩니다.

5. l 키를 누르면 도어락이 다시 잠기고 인증 정보가 초기화됩니다.

6. 종료하려면 l 키 누르고 Ctrl + C를 입력하세요.

---
