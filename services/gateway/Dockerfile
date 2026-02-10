# Build the Go gateway
FROM golang:1.24-alpine
WORKDIR /app
COPY . .
RUN go build -o gateway ./services/gateway/
EXPOSE 8080
CMD ["./gateway"]
