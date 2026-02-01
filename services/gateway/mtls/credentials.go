package mtls

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"

	"google.golang.org/grpc/credentials"
)

// Config holds mTLS configuration paths
type Config struct {
	CertFile   string // Path to server/client certificate
	KeyFile    string // Path to private key
	CAFile     string // Path to CA certificate for verification
	ServerName string // Expected server name for client connections
}

// LoadServerCredentials creates server-side mTLS credentials
func LoadServerCredentials(cfg Config) (credentials.TransportCredentials, error) {
	// Load server certificate and key
	serverCert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load server cert/key: %w", err)
	}

	// Load CA certificate for client verification
	caCert, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA cert: %w", err)
	}

	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA cert")
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{serverCert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    caPool,
		MinVersion:   tls.VersionTLS13,
	}

	return credentials.NewTLS(tlsConfig), nil
}

// LoadClientCredentials creates client-side mTLS credentials
func LoadClientCredentials(cfg Config) (credentials.TransportCredentials, error) {
	// Load client certificate and key
	clientCert, err := tls.LoadX509KeyPair(cfg.CertFile, cfg.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load client cert/key: %w", err)
	}

	// Load CA certificate for server verification
	caCert, err := os.ReadFile(cfg.CAFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA cert: %w", err)
	}

	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA cert")
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{clientCert},
		RootCAs:      caPool,
		ServerName:   cfg.ServerName,
		MinVersion:   tls.VersionTLS13,
	}

	return credentials.NewTLS(tlsConfig), nil
}

// LoadFromEnv creates mTLS config from environment variables
func LoadFromEnv(prefix string) Config {
	return Config{
		CertFile:   os.Getenv(prefix + "_CERT_FILE"),
		KeyFile:    os.Getenv(prefix + "_KEY_FILE"),
		CAFile:     os.Getenv(prefix + "_CA_FILE"),
		ServerName: os.Getenv(prefix + "_SERVER_NAME"),
	}
}

// MustLoadServerCredentials loads server credentials or panics
func MustLoadServerCredentials(cfg Config) credentials.TransportCredentials {
	creds, err := LoadServerCredentials(cfg)
	if err != nil {
		panic(fmt.Sprintf("failed to load mTLS server credentials: %v", err))
	}
	return creds
}

// MustLoadClientCredentials loads client credentials or panics
func MustLoadClientCredentials(cfg Config) credentials.TransportCredentials {
	creds, err := LoadClientCredentials(cfg)
	if err != nil {
		panic(fmt.Sprintf("failed to load mTLS client credentials: %v", err))
	}
	return creds
}
