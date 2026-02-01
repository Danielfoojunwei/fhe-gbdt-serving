# Build the Go registry
FROM golang:1.21-alpine
WORKDIR /app
COPY . .
RUN go build -o registry main.go
EXPOSE 8081
CMD ["./registry"]
