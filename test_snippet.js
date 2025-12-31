const CryptoJS = require('crypto-js');

const SECRET_KEY = "gQ9bX7pR2mKs";
const SECRET_HASH = "T8sJfQ2wLm9d";

function generateLoginURL(email = 'asingla.qs@gmail.com') {
    const expiry = Date.now() + 30 * 1000;  // 30 seconds url expiry

    const payload = JSON.stringify({
        email,
        expiry
    });

    // Encrypt
    const encrypted = CryptoJS.AES.encrypt(payload, SECRET_KEY).toString();

    // Encode for URL
    const encoded = encodeURIComponent(encrypted);
    const signature = CryptoJS.HmacSHA256(encrypted, SECRET_HASH).toString();

    // FIXED: Added backticks for template literal and 'const' declaration
    const redirect_url = `http://youngminds.pro/ymlab/user/login?token=${encodeURIComponent(encrypted)}&sig=${signature}`;

    console.log(redirect_url);
}

try {
    generateLoginURL();
} catch (e) {
    console.error("Error:", e);
}
