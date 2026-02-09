# Build the Go keystore
FROM golang:1.21-alpine
WORKDIR /app
COPY . .
RUN go build -o keystore ./services/keystore/
EXPOSE 8082
CMD ["./keystore"]
