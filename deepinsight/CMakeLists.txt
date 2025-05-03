cmake_minimum_required(VERSION 3.10)
project(FaceID)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 소스 파일
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(SOURCES_IDENTIFICATION ${SRC_DIR}/face_identification.cpp)
set(SOURCES_REGISTERATION ${SRC_DIR}/face_registeration.cpp)

# 라이브러리 및 include 경로
set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/lib/onnxruntime)
set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/lib/opencv)
set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session
    ${PROJECT_SOURCE_DIR}/include/opencv4
    ${PROJECT_SOURCE_DIR}/include/nlohmann
)

# 실행 파일 1: face_identification
add_executable(face_identification ${SOURCES_IDENTIFICATION})
target_include_directories(face_identification PRIVATE ${INCLUDE_DIRS})
target_link_directories(face_identification PRIVATE ${ONNXRUNTIME_DIR} ${OPENCV_DIR})
target_link_libraries(face_identification
    onnxruntime
    onnxruntime_providers_shared
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_dnn
)

# 실행 파일 2: face_registeration
add_executable(face_registeration ${SOURCES_REGISTERATION})
target_include_directories(face_registeration PRIVATE ${INCLUDE_DIRS})
target_link_directories(face_registeration PRIVATE ${ONNXRUNTIME_DIR} ${OPENCV_DIR})
target_link_libraries(face_registeration
    onnxruntime
    onnxruntime_providers_shared
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
    opencv_dnn
)

# OpenMP 병렬화 지원
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(face_identification OpenMP::OpenMP_CXX)
    target_link_libraries(face_registeration OpenMP::OpenMP_CXX)
endif()

# 빌드 타입 최적화
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(face_identification PRIVATE -O3)
    target_compile_options(face_registeration PRIVATE -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(face_identification PRIVATE -O0 -g)
    target_compile_options(face_registeration PRIVATE -O0 -g)
endif()

# 출력 바이너리 경로 설정
set_target_properties(face_identification PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)

set_target_properties(face_registeration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)
