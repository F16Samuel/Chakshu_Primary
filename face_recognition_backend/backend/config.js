// face/backend/config.js
require('dotenv').config(); // Load environment variables from .env file
const path = require('path');

const config = {
    port: process.env.PORT || 3001,
    dbPath: path.resolve(__dirname, process.env.DB_RELATIVE_PATH || '../campus_access.db'),
    recognitionApiUrl: process.env.RECOGNITION_API_URL || 'http://localhost:8001',
    registrationApiUrl: process.env.REGISTRATION_API_URL || 'http://localhost:8000',
    corsOrigins: process.env.CORS_ORIGINS ? process.env.CORS_ORIGINS.split(',') : ['*'],
    logLevel: process.env.LOG_LEVEL || 'info',
};

module.exports = config;