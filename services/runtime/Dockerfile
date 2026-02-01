# Build the C++ runtime with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y build-essential cmake
WORKDIR /app
COPY . .
# Standard CMake build (now checks for CUDA)
RUN mkdir build && cd build && cmake .. -DENABLE_GPU=ON && make -j
# RUN g++ -o runtime main.cpp (Replaced by CMake)
EXPOSE 50051
CMD ["./build/runtime"]
