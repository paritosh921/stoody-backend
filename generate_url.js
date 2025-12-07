const CryptoJS = require('crypto-js');

// Get arguments from command line
const args = process.argv.slice(2);
const email = args[0] || 'asingla.qs@gmail.com';
const secretKey = args[1] || "gQ9bX7pR2mKs";
const secretHash = args[2] || "T8sJfQ2wLm9d";

function generateLoginURL(email) {
    // 60 seconds expiry (Changed from 300s to match requirement closer, providing slight buffer)
    const expiry = Date.now() + 60 * 1000;

    const payload = JSON.stringify({
        email,
        expiry
    });

    // Encrypt
    const encrypted = CryptoJS.AES.encrypt(payload, secretKey).toString();

    // Generate signature
    const signature = CryptoJS.HmacSHA256(encrypted, secretHash).toString();

    // Construct Full URL exactly as requested
    const baseUrl = "http://youngminds.pro/ymlab/user/login";
    const redirect_url = `${baseUrl}?token=${encodeURIComponent(encrypted)}&sig=${signature}`;

    console.log(redirect_url);
}

try {
    generateLoginURL(email);
} catch (e) {
    console.error("Error:", e);
    process.exit(1);
}
