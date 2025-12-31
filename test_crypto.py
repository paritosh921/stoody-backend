"""
Test script to verify CryptoJS-compatible encryption
"""
import hashlib
import base64
import json
import time
import hmac
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

SECRET_KEY = 'gQ9bX7pR2mKs'
SECRET_HASH = 'T8sJfQ2wLm9d'

def evp_bytes_to_key(password, salt, key_len=32, iv_len=16):
    """OpenSSL EVP_BytesToKey derivation"""
    dtot = b''
    d = b''
    while len(dtot) < key_len + iv_len:
        d = hashlib.md5(d + password + salt).digest()
        dtot += d
    return dtot[:key_len], dtot[key_len:key_len + iv_len]

def encrypt_cryptojs(payload, passphrase):
    """Encrypt in CryptoJS-compatible format"""
    salt = secrets.token_bytes(8)
    key, iv = evp_bytes_to_key(passphrase.encode('utf-8'), salt)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(payload.encode('utf-8')) + padder.finalize()
    
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    openssl_format = b'Salted__' + salt + encrypted
    
    return base64.b64encode(openssl_format).decode('utf-8')

def decrypt_cryptojs(encrypted_b64, passphrase):
    """Decrypt CryptoJS-encrypted data"""
    data = base64.b64decode(encrypted_b64)
    
    # Check for "Salted__" prefix
    if data[:8] != b'Salted__':
        raise ValueError("Invalid format - missing 'Salted__' prefix")
    
    salt = data[8:16]
    ciphertext = data[16:]
    
    key, iv = evp_bytes_to_key(passphrase.encode('utf-8'), salt)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    decrypted_padded = decryptor.update(ciphertext) + decryptor.finalize()
    
    # Remove PKCS7 padding
    unpadder = padding.PKCS7(128).unpadder()
    decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
    
    return decrypted.decode('utf-8')

if __name__ == '__main__':
    # Test with example
    email = 'paritoshm921@gmail.com'
    expiry = int(time.time() * 1000) + (30 * 1000)
    payload = json.dumps({'email': email, 'expiry': expiry})
    print(f'Original Payload: {payload}')
    
    encrypted = encrypt_cryptojs(payload, SECRET_KEY)
    print(f'\nEncrypted (base64): {encrypted}')
    
    # Verify it starts with Salted__
    decoded = base64.b64decode(encrypted)
    print(f'Starts with Salted__: {decoded[:8] == b"Salted__"}')
    print(f'Salt (hex): {decoded[8:16].hex()}')
    
    # Test decryption to verify round-trip works
    decrypted = decrypt_cryptojs(encrypted, SECRET_KEY)
    print(f'\nDecrypted: {decrypted}')
    print(f'Round-trip successful: {decrypted == payload}')
    
    # Generate signature
    sig = hmac.new(SECRET_HASH.encode('utf-8'), encrypted.encode('utf-8'), hashlib.sha256).hexdigest()
    print(f'\nSignature: {sig}')
    
    # Build URL
    from urllib.parse import quote
    url = f"http://youngminds.pro/ymlab/user/login?token={quote(encrypted)}&sig={sig}"
    print(f'\nFull URL:\n{url}')
