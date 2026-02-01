# Build the Go gateway
FROM golang:1.21-alpine
WORKDIR /app
COPY . .
RUN go build -o gateway main.go
EXPOSE 8080
CMD ["./gateway"]
