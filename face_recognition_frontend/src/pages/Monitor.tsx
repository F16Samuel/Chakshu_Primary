// src/pages/monitor.tsx

import { useState, useEffect } from "react";
import { Users, BarChart3, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CameraFeed } from "@/components/camera/CameraFeed";
import { ActivityFeed } from "@/components/activity/ActivityFeed";
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Define TypeScript interfaces for better type safety
interface ActivityLog {
  id: string;
  userId: string;
  userName: string; // Ensure this is present
  action: "entry" | "exit" | "entry_attempt_already_on" | "exit_attempt_already_off"; // Added new actions for logging
  method: "scanner" | "kiosk_capture"; // Changed 'manual' to 'kiosk_capture' for clarity and consistency
  confidence?: number;
  timestamp: string; // <--- ADDED THIS LINE BACK IN!
}

interface PersonnelBreakdown {
  students: number;
  professors: number;
  guards: number;
  maintenance: number;
}

interface TodayStats {
  totalEntries: number; // Sum of entries + exits
  entries: number;
  exits: number;
  peakHour: string;
}

// Base URL for your FastAPI backend (for recognition endpoints)
// const FASTAPI_BASE_URL = "http://localhost:8001"; // No longer directly used by frontend
// Base URL for your Node.js backend (for data fetching and proxied recognition endpoints)
const API_BASE_URL = "http://localhost:3001/api"; // This is your Node.js backend for data and proxy

export default function Monitor() {
  // State to hold data fetched from the backend
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [personnelBreakdown, setPersonnelBreakdown] = useState<PersonnelBreakdown>({
    students: 0,
    professors: 0,
    guards: 0,
    maintenance: 0,
  });
  const [totalOnSite, setTotalOnSite] = useState(0); // This will need refinement for actual "on site" count
  const [todayStats, setTodayStats] = useState<TodayStats>({
    totalEntries: 0,
    entries: 0,
    exits: 0,
    peakHour: "N/A",
  });

  // Loading and error states for a better UX
  const [loadingActivities, setLoadingActivities] = useState(true);
  const [loadingPersonnel, setLoadingPersonnel] = useState(true); // Added 'useState(true)'
  const [loadingTodayStats, setLoadingTodayStats] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // States for image capture/recognition process
  const [isEntryCapturing, setIsEntryCapturing] = useState(false);
  const [isExitCapturing, setIsExitCapturing] = useState(false);
  const [recognitionMessage, setRecognitionMessage] = useState<string | null>(null);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);

  // Debugging: Log state changes for troubleshooting
  useEffect(() => {
    console.log("DEBUG: activities updated:", activities.length);
  }, [activities]);

  useEffect(() => {
    console.log("DEBUG: personnelBreakdown updated:", personnelBreakdown);
  }, [personnelBreakdown]);

  useEffect(() => {
    console.log("DEBUG: todayStats updated:", todayStats);
  }, [todayStats]);


  // Function to fetch total on-site personnel (if not derived from breakdown)
  const fetchTotalOnSite = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/total-on-site`); // Node.js backend
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setTotalOnSite(data.totalOnSite);
      console.log("DEBUG: Total on site fetched:", data.totalOnSite);
    } catch (err: any) {
      console.error("Error fetching total on site:", err);
      setFetchError(`Failed to load total on-site count: ${err.message}`);
    }
  };


  // Function to fetch recent activities
  const fetchActivities = async () => {
    console.log("DEBUG: Fetching activities...");
    setLoadingActivities(true);
    setFetchError(null); // Clear global fetch error before new attempt
    try {
      const response = await fetch(`${API_BASE_URL}/activities`); // Node.js backend
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: ActivityLog[] = await response.json();
      setActivities(data);
      console.log("DEBUG: Activities fetched successfully:", data.length, "items.");
    } catch (err: any) {
      console.error("Error fetching activities:", err);
      setFetchError(`Failed to load activities: ${err.message}`);
    } finally {
      setLoadingActivities(false);
    }
  };

  // Function to fetch personnel breakdown
  const fetchPersonnelBreakdown = async () => {
    console.log("DEBUG: Fetching personnel breakdown...");
    setLoadingPersonnel(true);
    setFetchError(null); // Clear global fetch error before new attempt
    try {
      const response = await fetch(`${API_BASE_URL}/personnel-breakdown`); // Node.js backend
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: PersonnelBreakdown = await response.json();
      setPersonnelBreakdown(data);
      fetchTotalOnSite(); // Fetch the actual total on-site count
      console.log("DEBUG: Personnel breakdown fetched:", data);
    } catch (err: any) {
      console.error("Error fetching personnel breakdown:", err);
      setFetchError(`Failed to load personnel data: ${err.message}`);
    } finally {
      setLoadingPersonnel(false);
    }
  };

  // Function to fetch today's statistics
  const fetchTodayStats = async () => {
    console.log("DEBUG: Fetching today's stats...");
    setLoadingTodayStats(true);
    setFetchError(null); // Clear global fetch error before new attempt
    try {
      const response = await fetch(`${API_BASE_URL}/today-stats`); // Node.js backend
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TodayStats = await response.json();
      setTodayStats(data);
      console.log("DEBUG: Today's stats fetched:", data);
    } catch (err: any) {
      console.error("Error fetching today's stats:", err);
      setFetchError(`Failed to load daily statistics: ${err.message}`);
    } finally {
      setLoadingTodayStats(false);
    }
  };

  // useEffect hook to fetch data when the component mounts and set up intervals
  useEffect(() => {
    console.log("DEBUG: Monitor component mounted. Initial data fetch and interval setup.");
    // Initial data fetch
    fetchActivities();
    fetchPersonnelBreakdown();
    fetchTotalOnSite(); // Ensure this is also fetched initially
    fetchTodayStats();

    // Set up intervals to refresh data periodically
    const activityInterval = setInterval(fetchActivities, 5000); // Every 5 seconds
    const personnelInterval = setInterval(fetchPersonnelBreakdown, 10000); // Every 10 seconds
    const statsInterval = setInterval(fetchTodayStats, 30000); // Every 30 seconds
    const totalOnSiteInterval = setInterval(fetchTotalOnSite, 10000); // Refresh total on site also

    // Cleanup intervals when the component unmounts
    return () => {
      console.log("DEBUG: Monitor component unmounting. Clearing intervals.");
      clearInterval(activityInterval);
      clearInterval(personnelInterval);
      clearInterval(statsInterval);
      clearInterval(totalOnSiteInterval);
    };
  }, []); // Empty dependency array ensures this runs once on mount

  // Helper to convert base64 image data to Blob for FormData submission
  const dataURLtoBlob = (dataurl: string) => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    console.log(`DEBUG: Converted data URL to Blob with type: ${mime}, size: ${u8arr.length} bytes`);
    return new Blob([u8arr], { type: mime });
  };

  // Handler for capturing an image for Entry recognition
  const handleCaptureEntry = async (imageData: string) => {
    console.log("DEBUG: Handling Entry Capture...");
    setIsEntryCapturing(true);
    setRecognitionMessage(null); // Clear previous messages
    setRecognitionError(null); // Clear previous errors
    try {
      if (!imageData) {
        throw new Error("No image data provided for entry capture.");
      }
      const blob = dataURLtoBlob(imageData);
      const formData = new FormData();
      formData.append("file", blob, "entry_capture.jpg"); // 'file' must match FastAPI's UploadFile parameter name
      console.log("DEBUG: FormData prepared for entry.");

      // CHANGED: Call Node.js backend proxy endpoint instead of direct FastAPI
      const response = await fetch(`${API_BASE_URL}/proxy/enter_site_recognition`, {
        method: 'POST',
        body: formData,
      });
      console.log(`DEBUG: Node.js /api/proxy/enter_site_recognition response status: ${response.status}`);

      const data = await response.json();
      console.log("DEBUG: Node.js response data for entry:", data);

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP error! status: ${response.status}`);
      }

      setRecognitionMessage(data.message || "Entry recognition successful!");
      console.log("INFO: Entry recognition successful:", data.message);
      // Immediately refresh relevant data to reflect the change
      fetchPersonnelBreakdown();
      fetchActivities();
      fetchTodayStats(); // Refresh stats after an entry

    } catch (err: any) {
      console.error("ERROR: during entry recognition:", err);
      setRecognitionError(`Entry recognition failed: ${err.message}. Please try again.`);
    } finally {
      setIsEntryCapturing(false);
      console.log("DEBUG: Finished handling entry capture.");
    }
  };

  // Handler for capturing an image for Exit recognition
  const handleCaptureExit = async (imageData: string) => {
    console.log("DEBUG: Handling Exit Capture...");
    setIsExitCapturing(true);
    setRecognitionMessage(null); // Clear previous messages
    setRecognitionError(null); // Clear previous errors
    try {
      if (!imageData) {
        throw new Error("No image data provided for exit capture.");
      }
      const blob = dataURLtoBlob(imageData);
      const formData = new FormData();
      formData.append("file", blob, "exit_capture.jpg"); // 'file' must match FastAPI's UploadFile parameter name
      console.log("DEBUG: FormData prepared for exit.");

      // CHANGED: Call Node.js backend proxy endpoint instead of direct FastAPI
      const response = await fetch(`${API_BASE_URL}/proxy/exit_site_recognition`, {
        method: 'POST',
        body: formData,
      });
      console.log(`DEBUG: Node.js /api/proxy/exit_site_recognition response status: ${response.status}`);

      const data = await response.json();
      console.log("DEBUG: Node.js response data for exit:", data);

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP error! status: ${response.status}`);
      }

      setRecognitionMessage(data.message || "Exit recognition successful!");
      console.log("INFO: Exit recognition successful:", data.message);
      // Immediately refresh relevant data
      fetchPersonnelBreakdown();
      fetchActivities();
      fetchTodayStats(); // Refresh stats after an exit

    } catch (err: any) {
      console.error("ERROR: during exit recognition:", err);
      setRecognitionError(`Exit recognition failed: ${err.message}. Please try again.`);
    } finally {
      setIsExitCapturing(false);
      console.log("DEBUG: Finished handling exit capture.");
    }
  };

  return (
    <div className="min-h-full bg-gradient-primary">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        {/* Header - This existing header content will now appear below TopButtons */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-gradient-neon bg-clip-text text-transparent mb-2">
            Campus Access Kiosk
          </h1>
          <p className="text-muted-foreground">
            Capture photos for campus entry and exit recognition
          </p>
        </div>

        {/* Global Error Display (for initial data fetches) */}
        {fetchError && (
          <div className="bg-red-500 text-white p-3 rounded-md mb-4">
            <p className="font-semibold">Data Loading Error:</p>
            <p>{fetchError}</p>
            <p className="text-sm mt-1">Please ensure both your FastAPI (8001) and Node.js (3001) backend servers are running correctly.</p>
          </div>
        )}

        {/* Recognition Status/Error Display (for capture actions) */}
        {recognitionMessage && (
          <div className="bg-green-600 text-white p-3 rounded-md mb-4 animate-fade-in-down">
            <p className="font-semibold">Success:</p>
            <p>{recognitionMessage}</p>
          </div>
        )}
        {recognitionError && (
          <div className="bg-red-500 text-white p-3 rounded-md mb-4 animate-fade-in-down">
            <p className="font-semibold">Recognition Error:</p>
            <p>{recognitionError}</p>
          </div>
        )}

        {/* Main Control Panel */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8 mb-8">
          {/* Camera Feeds (now local webcam capture) */}
          <div className="xl:col-span-3 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <CameraFeed
              title="ENTRY KIOSK"
              onCapture={handleCaptureEntry}
              loading={isEntryCapturing}
              // Disable entry if exit is currently capturing to avoid race conditions/errors
              disabled={isExitCapturing}
            />

            <CameraFeed
              title="EXIT KIOSK"
              onCapture={handleCaptureExit}
              loading={isExitCapturing}
              // Disable exit if entry is currently capturing
              disabled={isEntryCapturing}
            />
          </div>

          {/* Control Panel Sidebar */}
          <div className="space-y-6">
            {/* Personnel Count */}
            <Card className="bg-gradient-card border-primary/30 p-6">
              <h3 className="font-semibold mb-4 flex items-center space-x-2">
                <Users className="h-5 w-5 text-primary" />
                <span>Personnel On Site</span>
              </h3>

              <div className="text-center mb-4">
                {loadingPersonnel ? (
                  <p className="text-xl text-muted-foreground animate-pulse">Loading...</p>
                ) : (
                  <p className="text-3xl font-bold text-primary">{totalOnSite}</p>
                )}
                <p className="text-sm text-muted-foreground">Total Present</p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Students</span>
                  <Badge variant="secondary">{personnelBreakdown.students}</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Professors</span>
                  <Badge variant="secondary">{personnelBreakdown.professors}</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Guards</span>
                  <Badge variant="secondary">{personnelBreakdown.guards}</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Maintenance</span>
                  <Badge variant="secondary">{personnelBreakdown.maintenance}</Badge>
                </div>
              </div>
            </Card>

            {/* Today's Stats */}
            <Card className="bg-gradient-card border-primary/30 p-6">
              <h3 className="font-semibold mb-4 flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                <span>Today's Activity</span>
              </h3>

              <div className="space-y-4">
                <div className="text-center">
                  {loadingTodayStats ? (
                    <p className="text-xl text-muted-foreground animate-pulse">Loading...</p>
                  ) : (
                    <p className="text-2xl font-bold text-primary">{todayStats.totalEntries}</p>
                  )}
                  <p className="text-xs text-muted-foreground">Total Movements</p>
                </div>

                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    {loadingTodayStats ? (
                      <p className="text-lg text-muted-foreground animate-pulse">...</p>
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{todayStats.entries}</p>
                    )}
                    <p className="text-xs text-muted-foreground">Entries</p>
                  </div>
                  <div>
                    {loadingTodayStats ? (
                      <p className="text-lg text-muted-foreground animate-pulse">...</p>
                    ) : (
                      <p className="text-lg font-semibold text-foreground">{todayStats.exits}</p>
                    )}
                    <p className="text-xs text-muted-foreground">Exits</p>
                  </div>
                </div>

                <div className="pt-3 border-t border-border">
                  <div className="flex items-center justify-center space-x-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    <span>Peak: {todayStats.peakHour}</span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Activity Feed */}
        {loadingActivities ? (
          <p className="text-center text-muted-foreground animate-pulse">Loading activity feed...</p>
        ) : (
          <ActivityFeed activities={activities} className="h-96" />
        )}
      </div>
    </div>
  );
}