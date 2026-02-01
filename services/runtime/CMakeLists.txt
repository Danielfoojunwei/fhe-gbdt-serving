cmake_minimum_required(VERSION 3.18)
project(fhe_gbdt_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(ENABLE_GPU "Enable GPU backend" OFF)
option(ENABLE_TESTS "Enable unit tests" ON)
option(USE_SYSTEM_N2HE "Use system-installed N2HE" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# Try to find gRPC and Protobuf
find_package(gRPC CONFIG QUIET)
find_package(Protobuf CONFIG QUIET)

if(NOT gRPC_FOUND)
    message(STATUS "gRPC not found via find_package, will use pkg-config or manual setup")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_CURRENT_BINARY_DIR}/gen
)

# N2HE Integration
if(USE_SYSTEM_N2HE)
    find_package(n2he REQUIRED)
    set(N2HE_LIBRARIES n2he::n2he)
else()
    # Build vendored N2HE
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/n2he/CMakeLists.txt")
        add_subdirectory(third_party/n2he)
        set(N2HE_LIBRARIES n2he)
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/n2he/include)
    else()
        message(WARNING "Vendored N2HE not found, building without FHE support")
        add_definitions(-DNO_N2HE)
        set(N2HE_LIBRARIES "")
    endif()
endif()

# Core source files
set(RUNTIME_SOURCES
    main.cpp
    kernel/crypto_context.cpp
    kernel/step.cpp
    engine/executor.cpp
)

# GPU Backend
if(ENABLE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    list(APPEND RUNTIME_SOURCES 
        kernel/gpu/gpu_backend.cu
    )
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")  # Support Volta, Turing, Ampere
endif()

# Build runtime executable
add_executable(runtime ${RUNTIME_SOURCES})

target_link_libraries(runtime PRIVATE
    ${N2HE_LIBRARIES}
    Threads::Threads
)

if(ENABLE_GPU)
    target_compile_definitions(runtime PRIVATE ENABLE_GPU)
    target_link_libraries(runtime PRIVATE 
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()

if(gRPC_FOUND)
    target_link_libraries(runtime PRIVATE
        gRPC::grpc++
        protobuf::libprotobuf
    )
endif()

# Install
install(TARGETS runtime DESTINATION bin)

# Tests
if(ENABLE_TESTS)
    enable_testing()
    
    # Add test executable
    add_executable(runtime_tests
        tests/test_cpu_backend.cpp
        tests/test_crypto_context.cpp
    )
    
    target_link_libraries(runtime_tests PRIVATE
        ${N2HE_LIBRARIES}
        Threads::Threads
    )
    
    add_test(NAME RuntimeTests COMMAND runtime_tests)
endif()

