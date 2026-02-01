# Build the Go keystore
FROM golang:1.21-alpine
WORKDIR /app
COPY . .
RUN go build -o keystore main.go
EXPOSE 8082
CMD ["./keystore"]
