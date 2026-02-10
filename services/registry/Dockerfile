# Build the Go registry
FROM golang:1.24-alpine
WORKDIR /app
COPY . .
RUN go build -o registry ./services/registry/
EXPOSE 8081
CMD ["./registry"]
