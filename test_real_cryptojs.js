// Test using the ACTUAL CryptoJS library to generate a URL
// This should be 100% compatible with what YoungMinds expects

const CryptoJS = require('crypto-js');

const SECRET_KEY = "gQ9bX7pR2mKs";
const SECRET_HASH = "T8sJfQ2wLm9d";

function generateLoginURL(email = 'asingla.qs@gmail.com') {
    const expiry = Date.now() + 300 * 1000;  // 5 minutes url expiry (changed from 30s)

    const payload = JSON.stringify({
        email,
        expiry
    });

    console.log('Payload:', payload);

    // Encrypt using actual CryptoJS
    const encrypted = CryptoJS.AES.encrypt(payload, SECRET_KEY).toString();
    console.log('Encrypted:', encrypted);

    // Generate signature
    const signature = CryptoJS.HmacSHA256(encrypted, SECRET_HASH).toString();
    console.log('Signature:', signature);

    // Build URL
    const redirect_url = `http://youngminds.pro/ymlab/user/login?token=${encodeURIComponent(encrypted)}&sig=${signature}`;
    console.log('\nFull URL:');
    console.log(redirect_url);

    return redirect_url;
}

// Generate and print URL
generateLoginURL();
