package main

import (
	"log"
	"net/http"
)

func main() {
	log.Println("Keystore service starting...")
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})
	log.Fatal(http.ListenAndServe(":8082", nil))
}
