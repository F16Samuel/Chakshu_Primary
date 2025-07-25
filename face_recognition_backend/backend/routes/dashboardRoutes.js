// face/backend/routes/dashboardRoutes.js
const express = require('express');
const { dbAll, dbGet } = require('../db'); // Import database helpers

const router = express.Router();

// Get all activities (recent logs)
router.get('/activities', async (req, res) => {
    console.log("DEBUG: Received request for /api/activities");
    try {
        const activities = await dbAll(`
            SELECT
                al.id,
                al.user_id,
                u.name AS userName, -- Get user name from the users table
                al.action,
                al.timestamp,
                al.method,
                al.confidence
            FROM
                access_logs al
            JOIN
                users u ON al.user_id = u.id_number
            ORDER BY
                al.timestamp DESC
            LIMIT 50
        `);
        console.log(`DEBUG: Fetched ${activities.length} activities.`);
        res.json(activities);
    } catch (err) {
        console.error('ERROR: Error fetching activities:', err);
        res.status(500).json({ error: 'Failed to fetch activities' });
    }
});

// Get personnel breakdown by role
router.get('/personnel-breakdown', async (req, res) => {
    console.log("DEBUG: Received request for /api/personnel-breakdown");
    try {
        const students = await dbGet("SELECT COUNT(*) as count FROM users WHERE role = 'student' AND on_site = 1");
        const professors = await dbGet("SELECT COUNT(*) as count FROM users WHERE role = 'professor' AND on_site = 1");
        const guards = await dbGet("SELECT COUNT(*) as count FROM users WHERE role = 'guard' AND on_site = 1");
        const maintenance = await dbGet("SELECT COUNT(*) as count FROM users WHERE role = 'maintenance' AND on_site = 1");

        const breakdown = {
            students: students ? students.count : 0,
            professors: professors ? professors.count : 0,
            guards: guards ? guards.count : 0,
            maintenance: maintenance ? maintenance.count : 0,
        };
        console.log("DEBUG: Personnel breakdown:", breakdown);
        res.json(breakdown);
    } catch (err) {
        console.error('ERROR: Error fetching personnel breakdown:', err);
        res.status(500).json({ error: 'Failed to fetch personnel breakdown' });
    }
});

// Get total personnel currently on site
router.get('/total-on-site', async (req, res) => {
    console.log("DEBUG: Received request for /api/total-on-site");
    try {
        const total = await dbGet("SELECT COUNT(*) as count FROM users WHERE on_site = 1");
        console.log("DEBUG: Total on site:", total ? total.count : 0);
        res.json({ totalOnSite: total ? total.count : 0 });
    } catch (err) {
        console.error('ERROR: Error fetching total on site:', err);
        res.status(500).json({ error: 'Failed to fetch total on site' });
    }
});

// Get today's activity statistics
router.get('/today-stats', async (req, res) => {
    console.log("DEBUG: Received request for /api/today-stats");
    try {
        const startOfDay = new Date();
        startOfDay.setHours(0, 0, 0, 0);
        const endOfDay = new Date();
        endOfDay.setHours(23, 59, 59, 999);

        const startOfDayISO = startOfDay.toISOString(); // Assuming timestamps are ISO strings
        const endOfDayISO = endOfDay.toISOString();

        const entries = await dbGet("SELECT COUNT(*) as count FROM access_logs WHERE action = 'entry' AND timestamp BETWEEN ? AND ?", [startOfDayISO, endOfDayISO]);
        const exits = await dbGet("SELECT COUNT(*) as count FROM access_logs WHERE action = 'exit' AND timestamp BETWEEN ? AND ?", [startOfDayISO, endOfDayISO]);
        const total = (entries?.count || 0) + (exits?.count || 0);

        // Find peak hour (example: group by hour and find the busiest hour)
        const peakHourResult = await dbGet(`
            SELECT STRFTIME('%H', timestamp) as hour, COUNT(*) as count
            FROM access_logs
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY hour
            ORDER BY count DESC
            LIMIT 1
        `, [startOfDayISO, endOfDayISO]);

        const peakHour = peakHourResult ? `${peakHourResult.hour}:00-${parseInt(peakHourResult.hour) + 1}:00` : 'N/A';
        console.log("DEBUG: Today's stats:", { totalEntries: total, entries: entries?.count || 0, exits: exits?.count || 0, peakHour });

        res.json({
            totalEntries: total, // Total movements (entries + exits)
            entries: entries?.count || 0,
            exits: exits?.count || 0,
            peakHour: peakHour,
        });
    } catch (err) {
        console.error("ERROR: Error fetching today's stats:", err);
        res.status(500).json({ error: "Failed to fetch today's stats" });
    }
});

module.exports = router;