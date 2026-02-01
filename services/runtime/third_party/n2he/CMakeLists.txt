cmake_minimum_required(VERSION 3.00)

project(n2he LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funroll-loops -fomit-frame-pointer -march=native -mtune=native")

# Define N2HE as an interface library (since it's header-only)
add_library(n2he INTERFACE)
target_include_directories(n2he INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(n2he INTERFACE pthread crypto)

# Keep the user's executables as optional
option(BUILD_N2HE_EXAMPLES "Build N2HE example executables" ON)
if(BUILD_N2HE_EXAMPLES)
    add_executable(n2he_test test.cpp)
    target_link_libraries(n2he_test n2he)

    add_executable(luttest LUT_test.cpp)
    target_link_libraries(luttest n2he)

    add_executable(face FacialRecognition.cpp)
    target_link_libraries(face n2he)

    add_executable(mnist MNIST.cpp)
    target_link_libraries(mnist n2he)

    add_executable(object ObjectClassification.cpp)
    target_link_libraries(object n2he)

    add_executable(speech SpeechRecognition.cpp)
    target_link_libraries(speech n2he)

    add_executable(text TextClassification.cpp)
    target_link_libraries(text n2he)
endif()
