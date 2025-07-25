// face/backend/db.js
const sqlite3 = require('sqlite3').verbose();
const config = require('./config'); // Import config to get dbPath

const db = new sqlite3.Database(config.dbPath, sqlite3.OPEN_READONLY, (err) => {
    if (err) {
        console.error('Could not connect to database:', err.message);
        // In a real application, you might want to exit the process or handle this more gracefully.
        // For now, we'll just log the error.
    } else {
        console.log('Connected to SQLite database at', config.dbPath);
    }
});

// Helper functions to promisify sqlite3 methods
const dbAll = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        db.all(sql, params, (err, rows) => {
            if (err) {
                console.error("Database dbAll error:", err.message, "SQL:", sql, "Params:", params);
                reject(err);
            } else {
                resolve(rows);
            }
        });
    });
};

const dbGet = (sql, params = []) => {
    return new Promise((resolve, reject) => {
        db.get(sql, params, (err, row) => {
            if (err) {
                console.error("Database dbGet error:", err.message, "SQL:", sql, "Params:", params);
                reject(err);
            } else {
                resolve(row);
            }
        });
    });
};

// Close the database connection gracefully
const closeDb = () => {
    return new Promise((resolve, reject) => {
        db.close((err) => {
            if (err) {
                console.error('Error closing database:', err.message);
                reject(err);
            } else {
                console.log('Closed the database connection.');
                resolve();
            }
        });
    });
};

module.exports = {
    dbAll,
    dbGet,
    closeDb
};