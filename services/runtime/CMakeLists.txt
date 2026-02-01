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
    engine/executor.cpp
)

# target_link_libraries(runtime 
#     n2he::n2he
#     hexl::hexl
#     gRPC::grpc++
#     protobuf::libprotobuf
# )
