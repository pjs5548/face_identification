1. 아래의 패키지들을 설치 해 주시길 바랍니다.
-------------------------------------------------------------------------------------
sudo apt update
sudo apt install -y \
    build-essential cmake pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
    zlib1g-dev libprotobuf-dev protobuf-compiler \
    nlohmann-json-dev libomp-dev libpthread-stubs0-dev \
    alsa-utils \
    python3 python3-pip python3-pyqt5 libgl1

# Python 패키지
pip install numpy==1.26.4 opencv-python==4.11.0 pyqt5==5.15.11

-------------------------------------------------------------------------------------

2. 얼굴 등록 절차
(1) python 폴더에 접속하셔서 python3 gui.py --port xxxx (xxxx는 9000~9999 사이의 아무 숫자나 입력) 를 입력하시면  Face Registeration Manager라는 이름의 GUI 창이 하나 뜹니다.
(2) 화면 중간에 보이는 부분이 이미 등록된 얼굴들의 임베딩 벡터를 raw 파일로 저장한것입니다.
(3) 각 임베딩 raw 파일을  클릭 후 remove를 누르시면 해당 raw 파일을 지울 수 있습니다.
(4) Remove all을 누르면 모든 raw 파일이 지워집니다.
(5) 얼굴을 등록하고 싶으시면 registeration 버튼을 눌러 주시길 바랍니다. 그러면 이니셜을 입력하라는 창이 나오고 그때 이니셜을 입력하고 확인 버튼을 눌러 주시면 주시면  10초간 기다리라는 창이 나오고 10초 뒤에 얼굴 바운딩 박스가 쳐진 화면이 나옵니다. 
(6) 얼굴 등록을 마무리 하고 싶으시다면 Face Registeration Manager 창의 우측 상단 x표를 클릭 하시면 종료 됩니다.
(7) 잠시 후 ssets/face_information 폴더에 이니셜.raw 형태로 얼굴 임베딩이 저장된걸 확인 하질 수 있습니다.

3. 얼굴 인증 절차
(1) bin 폴더에 접속하셔서 ./face_identifiaction --cam 0 --pot xxxx (xxxx는 9000~9999 사이의 아무 숫자나 입력) 를 입력하시면, 얼굴 인증 절차가 시작됩니다.
(2) u를 누르면 도어락이 풀려 누군가가 차에 탄것으로 인식이 되고 얼굴 인식이 시작됩니다.
(3) 1차로 얼굴 인식을 시작하고 통과되면 그대로 pass이며 통과 되지 못하면 5초뒤 얼굴 인식을 다시 시작합니다.
(4) pass 되었을시 터미널에 인식 정보가 표시되고 imshow를 통해 출력되는 창에도 이니셜이 표시가 됩니다.
(5) l을 누르면 다시 도어락이 잠기고 인증 정보가 초기화 됩니다.
(6) 종료를 하고 싶으시면 l을 누르고 ctrl+c를 누르시길 바랍니다.

