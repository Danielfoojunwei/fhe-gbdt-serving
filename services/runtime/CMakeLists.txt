cmake_minimum_required(VERSION 3.18)
project(fhe_gbdt_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies (Assuming they are available in the build environment/Docker)
# find_package(n2he REQUIRED)
# find_package(hexl REQUIRED)
# find_package(gRPC REQUIRED)
# find_package(Protobuf REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/proto
    gen # For generated gRPC stubs
)

add_executable(runtime 
    main.cpp
    kernel/crypto_context.cpp
    kernel/step.cpp
    kernel/step.cpp
    engine/executor.cpp
)

option(ENABLE_GPU "Enable GPU backend" OFF)

if(ENABLE_GPU)
    enable_language(CUDA)
    target_sources(runtime PRIVATE 
        kernel/gpu/gpu_backend.cu 
        kernel/gpu/gpu_backend.cpp # if any
    )
    target_compile_definitions(runtime PRIVATE ENABLE_GPU)
    # target_link_libraries(runtime PRIVATE CUDA::cudart)
endif()

# target_link_libraries(runtime 
#     n2he::n2he
#     hexl::hexl
#     gRPC::grpc++
#     protobuf::libprotobuf
# )

# Integration of Vendored N2HE
add_subdirectory(third_party/n2he)
target_link_libraries(runtime PRIVATE n2he)
