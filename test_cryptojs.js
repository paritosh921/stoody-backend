// Test CryptoJS encryption to compare with Python output
// Run with: node test_cryptojs.js

const crypto = require('crypto');

const SECRET_KEY = "gQ9bX7pR2mKs";
const SECRET_HASH = "T8sJfQ2wLm9d";

// EVP_BytesToKey implementation (what CryptoJS uses internally)
function evpBytesToKey(password, salt, keyLen = 32, ivLen = 16) {
    let derivedBytes = Buffer.alloc(0);
    let block = Buffer.alloc(0);

    while (derivedBytes.length < keyLen + ivLen) {
        const hash = crypto.createHash('md5');
        hash.update(block);
        hash.update(Buffer.from(password, 'utf-8'));
        hash.update(salt);
        block = hash.digest();
        derivedBytes = Buffer.concat([derivedBytes, block]);
    }

    return {
        key: derivedBytes.slice(0, keyLen),
        iv: derivedBytes.slice(keyLen, keyLen + ivLen)
    };
}

function encryptCryptoJSStyle(plaintext, passphrase) {
    // Random 8-byte salt
    const salt = crypto.randomBytes(8);

    // Derive key and IV
    const { key, iv } = evpBytesToKey(passphrase, salt);

    // Encrypt with AES-256-CBC
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    let encrypted = cipher.update(plaintext, 'utf-8');
    encrypted = Buffer.concat([encrypted, cipher.final()]);

    // OpenSSL format: "Salted__" + salt + ciphertext
    const result = Buffer.concat([
        Buffer.from('Salted__', 'utf-8'),
        salt,
        encrypted
    ]);

    return result.toString('base64');
}

function decryptCryptoJSStyle(encryptedB64, passphrase) {
    const data = Buffer.from(encryptedB64, 'base64');

    // Check for "Salted__" prefix
    const prefix = data.slice(0, 8).toString('utf-8');
    if (prefix !== 'Salted__') {
        throw new Error('Invalid format - missing Salted__ prefix');
    }

    const salt = data.slice(8, 16);
    const ciphertext = data.slice(16);

    // Derive key and IV
    const { key, iv } = evpBytesToKey(passphrase, salt);

    // Decrypt
    const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
    let decrypted = decipher.update(ciphertext);
    decrypted = Buffer.concat([decrypted, decipher.final()]);

    return decrypted.toString('utf-8');
}

function hmacSha256(data, secret) {
    return crypto.createHmac('sha256', secret).update(data).digest('hex');
}

// Test
const email = 'paritoshm921@gmail.com';
const expiry = Date.now() + 30 * 1000;
const payload = JSON.stringify({ email, expiry });

console.log('Original Payload:', payload);
console.log('');

const encrypted = encryptCryptoJSStyle(payload, SECRET_KEY);
console.log('Encrypted (base64):', encrypted);

// Verify round-trip
const decrypted = decryptCryptoJSStyle(encrypted, SECRET_KEY);
console.log('Decrypted:', decrypted);
console.log('Round-trip match:', decrypted === payload);
console.log('');

// Generate signature
const sig = hmacSha256(encrypted, SECRET_HASH);
console.log('Signature:', sig);

// Build URL
const url = `http://youngminds.pro/ymlab/user/login?token=${encodeURIComponent(encrypted)}&sig=${sig}`;
console.log('\nFull URL:');
console.log(url);
