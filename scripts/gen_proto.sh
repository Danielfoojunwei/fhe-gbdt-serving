#!/bin/bash
# Script to generate Protobuf stubs for Go, Python, and C++

set -e

PROTO_DIR="./fhe-gbdt-serving/proto"
SERVICES_DIR="./fhe-gbdt-serving/services"
SDK_DIR="./fhe-gbdt-serving/sdk"

echo "Generating Go stubs..."
# Gateway
protoc --proto_path=$PROTO_DIR \
       --go_out=$SERVICES_DIR/gateway/gen --go_opt=paths=source_relative \
       --go-grpc_out=$SERVICES_DIR/gateway/gen --go-grpc_opt=paths=source_relative \
       $PROTO_DIR/*.proto

# Registry
protoc --proto_path=$PROTO_DIR \
       --go_out=$SERVICES_DIR/registry/gen --go_opt=paths=source_relative \
       --go-grpc_out=$SERVICES_DIR/registry/gen --go-grpc_opt=paths=source_relative \
       $PROTO_DIR/*.proto

# Keystore
protoc --proto_path=$PROTO_DIR \
       --go_out=$SERVICES_DIR/keystore/gen --go_opt=paths=source_relative \
       --go-grpc_out=$SERVICES_DIR/keystore/gen --go-grpc_opt=paths=source_relative \
       $PROTO_DIR/*.proto

echo "Generating Python stubs..."
python -m grpc_tools.protoc --proto_path=$PROTO_DIR \
       --python_out=$SERVICES_DIR/compiler/gen \
       --grpc_python_out=$SERVICES_DIR/compiler/gen \
       $PROTO_DIR/*.proto

python -m grpc_tools.protoc --proto_path=$PROTO_DIR \
       --python_out=$SDK_DIR/python/gen \
       --grpc_python_out=$SDK_DIR/python/gen \
       $PROTO_DIR/*.proto

echo "Generating C++ stubs..."
protoc --proto_path=$PROTO_DIR \
       --cpp_out=$SERVICES_DIR/runtime/gen \
       --grpc_out=$SERVICES_DIR/runtime/gen \
       --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` \
       $PROTO_DIR/*.proto

echo "Done."
