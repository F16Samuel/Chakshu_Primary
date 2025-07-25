// face/backend/routes/proxyRoutes.js
const express = require('express');
const axios = require('axios'); // For making HTTP requests to Python services
const config = require('../config'); // Import config to get Python API URLs

const router = express.Router();

// --- Proxy Endpoints to Python Recognition Service ---

// Helper function to proxy requests
const proxyRequest = async (req, res, pythonServiceUrl, endpoint) => {
    const url = `${pythonServiceUrl}${endpoint}`;
    try {
        let response;
        if (req.method === 'GET') {
            // For GET requests, ensure params are passed
            response = await axios.get(url, { params: req.query });
        } else if (req.method === 'POST') {
            // For POST requests, ensure body is passed.
            // axios automatically handles JSON bodies.
            // For multipart/form-data, you might need specific handling (e.g., using 'form-data' library)
            // if you intend to re-proxy file uploads.
            response = await axios.post(url, req.body, {
                headers: {
                    // Crucial to preserve Content-Type for correct parsing by the FastAPI server,
                    // especially for file uploads that might come as multipart/form-data
                    'Content-Type': req.headers['content-type']
                },
                // If you expect binary data or streams back, adjust responseType
                responseType: 'arraybuffer' in req.headers ? 'arraybuffer' : 'json' // Example: if expecting binary image/video
            });
        } else {
            return res.status(405).json({ error: 'Method Not Allowed' });
        }

        // Forward headers and status from the proxied response
        for (const key in response.headers) {
            if (response.headers.hasOwnProperty(key)) {
                res.setHeader(key, response.headers[key]);
            }
        }
        res.status(response.status).send(response.data); // Use .send() for general data, .json() for JSON only

    } catch (error) {
        console.error(`Proxy Error to ${url}:`, error.message);
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
            // Attempt to send back original error status and data from Python service
            // Convert buffer to string if it's a binary error response
            let errorData = error.response.data;
            if (Buffer.isBuffer(errorData)) {
                try {
                    errorData = JSON.parse(errorData.toString());
                } catch (e) {
                    errorData = errorData.toString(); // Fallback to plain string
                }
            }
            res.status(error.response.status).send(errorData);
        } else {
            // Network error or no response from Python service
            res.status(500).json({ error: `Failed to connect to Python service: ${error.message}` });
        }
    }
};

// Recognition Service Endpoints
router.get('/recognition/health', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/health'));
router.get('/recognition/logs', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/logs'));
router.get('/recognition/users/:user_id/status', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, `/users/${req.params.user_id}/status`));
router.post('/recognition/users/:user_id/manual_entry', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, `/users/${req.params.user_id}/manual_entry`));
router.post('/recognition/users/:user_id/manual_exit', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, `/users/${req.params.user_id}/manual_exit`));
router.get('/recognition/reload-faces', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/reload-faces'));

// NEW Proxy Endpoints for Recognition Server (Scanner Control & Kiosk)
// Assuming these are simple GET/POST with JSON or no body.
// If /entry_video_feed or /exit_video_feed return multipart/x-mixed-replace,
// the proxyRequest function handles it by setting Content-Type and piping.
router.get('/recognition/scanner_status', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/scanner_status'));
router.post('/recognition/start_entry_scanning', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/start_entry_scanning'));
router.post('/recognition/stop_entry_scanning', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/stop_entry_scanning'));
router.post('/recognition/start_exit_scanning', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/start_exit_scanning'));
router.post('/recognition/stop_exit_scanning', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/stop_exit_scanning'));

router.get('/recognition/entry_video_feed', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/entry_video_feed'));
router.get('/recognition/exit_video_feed', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/exit_video_feed'));


// For file uploads like enter_site_recognition and exit_site_recognition,
// if they are multipart/form-data, the proxyRequest will need 'form-data' library or similar
// to re-construct the multipart request. For now, it assumes JSON body.
router.post('/recognition/enter_site_recognition', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/enter_site_recognition'));
router.post('/recognition/exit_site_recognition', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/exit_site_recognition'));
router.get('/recognition/on_site_personnel', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, '/on_site_personnel'));
router.get('/recognition/users/:role', (req, res) => proxyRequest(req, res, config.recognitionApiUrl, `/users/${req.params.role}`));


// --- Proxy Endpoints to Python Registration Service ---
router.get('/registration/health', (req, res) => proxyRequest(req, res, config.registrationApiUrl, '/health'));
router.post('/registration/register', (req, res) => proxyRequest(req, res, config.registrationApiUrl, '/register'));
router.post('/registration/webcam-capture', (req, res) => proxyRequest(req, res, config.registrationApiUrl, '/webcam-capture'));
router.get('/registration/users/:user_id', (req, res) => proxyRequest(req, res, config.registrationApiUrl, `/users/${req.params.user_id}`));
router.get('/registration/users', (req, res) => proxyRequest(req, res, config.registrationApiUrl, '/users'));


module.exports = router;