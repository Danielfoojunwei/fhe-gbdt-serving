package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"errors"
	"io"
)

// EnvelopeEncrypt encrypts data with a data encryption key (DEK),
// then encrypts the DEK with a key encryption key (KEK).
// Returns: encryptedDEK || nonce || ciphertext
func EnvelopeEncrypt(plaintext []byte, kek []byte) ([]byte, error) {
	// Generate random DEK
	dek := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, dek); err != nil {
		return nil, err
	}
	
	// Encrypt plaintext with DEK
	block, err := aes.NewCipher(dek)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}
	ciphertext := gcm.Seal(nil, nonce, plaintext, nil)
	
	// Encrypt DEK with KEK
	kekBlock, err := aes.NewCipher(kek)
	if err != nil {
		return nil, err
	}
	kekGCM, err := cipher.NewGCM(kekBlock)
	if err != nil {
		return nil, err
	}
	kekNonce := make([]byte, kekGCM.NonceSize())
	if _, err := io.ReadFull(rand.Reader, kekNonce); err != nil {
		return nil, err
	}
	encryptedDEK := kekGCM.Seal(nil, kekNonce, dek, nil)
	
	// Combine: kekNonce || encryptedDEK || nonce || ciphertext
	result := append(kekNonce, encryptedDEK...)
	result = append(result, nonce...)
	result = append(result, ciphertext...)
	return result, nil
}

// EnvelopeDecrypt decrypts envelope-encrypted data
func EnvelopeDecrypt(envelope []byte, kek []byte) ([]byte, error) {
	if len(envelope) < 64 {
		return nil, errors.New("envelope too short")
	}
	
	kekBlock, _ := aes.NewCipher(kek)
	kekGCM, _ := cipher.NewGCM(kekBlock)
	
	kekNonceSize := kekGCM.NonceSize()
	kekNonce := envelope[:kekNonceSize]
	
	// DEK is 32 bytes + 16 bytes GCM tag
	encryptedDEK := envelope[kekNonceSize : kekNonceSize+48]
	dek, err := kekGCM.Open(nil, kekNonce, encryptedDEK, nil)
	if err != nil {
		return nil, err
	}
	
	// Decrypt ciphertext with DEK
	dataPortion := envelope[kekNonceSize+48:]
	block, _ := aes.NewCipher(dek)
	gcm, _ := cipher.NewGCM(block)
	nonceSize := gcm.NonceSize()
	nonce := dataPortion[:nonceSize]
	ciphertext := dataPortion[nonceSize:]
	
	return gcm.Open(nil, nonce, ciphertext, nil)
}
