// backend/server.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const config = require('./config'); // Import the new config module
const { closeDb } = require('./db'); // Import db close function
const dashboardRoutes = require('./routes/dashboardRoutes');
const proxyRoutes = require('./routes/proxyRoutes');

const app = express();

// Enable CORS for all routes, allowing your React app to access it
// Use config.corsOrigins for more specific control
app.use(cors({ origin: config.corsOrigins }));
app.use(express.json()); // For parsing application/json
app.use(express.urlencoded({ extended: true })); // For parsing application/x-www-form-urlencoded

// Serve static files from the 'public' directory (assuming it's sibling to 'backend')
app.use(express.static(path.join(__dirname, '../public')));

// Use modularized routes
app.use('/api', dashboardRoutes); // For dashboard-specific data from Node.js's DB connection
app.use('/api', proxyRoutes);     // For all proxied requests to Python microservices

// Root endpoint for the Node.js server itself (optional, if not serving a frontend here)
app.get('/', (req, res) => {
    res.send('Node.js Backend API is running. Access dashboard via /public/index.html or your frontend app.');
});


app.listen(config.port, () => {
    console.log(`Node.js Backend API running on http://localhost:${config.port}`);
});

// Close the database connection gracefully on process termination
process.on('SIGINT', async () => {
    console.log('SIGINT signal received: Closing database and exiting.');
    try {
        await closeDb();
        process.exit(0);
    } catch (err) {
        console.error('Error during graceful shutdown:', err);
        process.exit(1);
    }
});

process.on('SIGTERM', async () => {
    console.log('SIGTERM signal received: Closing database and exiting.');
    try {
        await closeDb();
        process.exit(0);
    } catch (err) {
        console.error('Error during graceful shutdown:', err);
        process.exit(1);
    }
});