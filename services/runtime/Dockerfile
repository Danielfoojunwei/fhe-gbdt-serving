# Build the C++ runtime
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential cmake
WORKDIR /app
COPY . .
RUN g++ -o runtime main.cpp
EXPOSE 50051
CMD ["./runtime"]
