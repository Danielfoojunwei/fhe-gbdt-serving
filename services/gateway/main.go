package main

import (
	"net/http"
)

// registerHealthEndpoint adds a basic HTTP health check endpoint.
func registerHealthEndpoint() {
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})
}
