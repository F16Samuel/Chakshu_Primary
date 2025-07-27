// src/pages/monitor.tsx

import { useState, useEffect, useCallback } from "react";
import { Users, BarChart3, Clock } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CameraFeed } from "@/components/camera/CameraFeed"; // Keep if still using local webcam for manual capture
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
  timestamp: string;
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

// Base URL for your unified FastAPI backend
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

export default function Monitor() {
  // State to hold data fetched from the backend
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [personnelBreakdown, setPersonnelBreakdown] = useState<PersonnelBreakdown>({
    students: 0,
    professors: 0,
    guards: 0,
    maintenance: 0,
  });
  const [totalOnSite, setTotalOnSite] = useState(0);
  const [todayStats, setTodayStats] = useState<TodayStats>({
    totalEntries: 0,
    entries: 0,
    exits: 0,
    peakHour: "N/A",
  });

  // Loading and error states for a better UX
  const [loadingActivities, setLoadingActivities] = useState(true);
  const [loadingPersonnel, setLoadingPersonnel] = useState(true);
  const [loadingTodayStats, setLoadingTodayStats] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // States for image capture/recognition process (if still needed for manual capture on this page)
  const [isEntryCapturing, setIsEntryCapturing] = useState(false);
  const [isExitCapturing, setIsExitCapturing] = useState(false);
  const [recognitionMessage, setRecognitionMessage] = useState<string | null>(null);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);

  // Function to fetch total on-site personnel
  const fetchTotalOnSite = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/dashboard/total-on-site`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setTotalOnSite(data.totalOnSite);
    } catch (err: any) {
      console.error("Error fetching total on site:", err);
      setFetchError(`Failed to load total on-site count: ${err.message}`);
    }
  }, []);


  // Function to fetch recent activities
  const fetchActivities = useCallback(async () => {
    setLoadingActivities(true);
    setFetchError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/dashboard/activities`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: ActivityLog[] = await response.json();
      // Adjust 'userName' to 'user_name' to match FastAPI backend
      const transformedData = data.map(activity => ({
        ...activity,
        userName: (activity as any).user_name || activity.userName // Use user_name if available
      }));
      setActivities(transformedData);
    } catch (err: any) {
      console.error("Error fetching activities:", err);
      setFetchError(`Failed to load activities: ${err.message}`);
    } finally {
      setLoadingActivities(false);
    }
  }, []);

  // Function to fetch personnel breakdown
  const fetchPersonnelBreakdown = useCallback(async () => {
    setLoadingPersonnel(true);
    setFetchError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/dashboard/personnel-breakdown`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: PersonnelBreakdown = await response.json();
      setPersonnelBreakdown(data);
    } catch (err: any) {
      console.error("Error fetching personnel breakdown:", err);
      setFetchError(`Failed to load personnel data: ${err.message}`);
    } finally {
      setLoadingPersonnel(false);
    }
  }, []);

  // Function to fetch today's statistics
  const fetchTodayStats = useCallback(async () => {
    setLoadingTodayStats(true);
    setFetchError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/dashboard/today-stats`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TodayStats = await response.json();
      setTodayStats(data);
    } catch (err: any) {
      console.error("Error fetching today's stats:", err);
      setFetchError(`Failed to load daily statistics: ${err.message}`);
    } finally {
      setLoadingTodayStats(false);
    }
  }, []);

  // useEffect hook to fetch data when the component mounts and set up intervals
  useEffect(() => {
    fetchActivities();
    fetchPersonnelBreakdown();
    fetchTotalOnSite();
    fetchTodayStats();

    const activityInterval = setInterval(fetchActivities, 5000); // Every 5 seconds
    const personnelInterval = setInterval(fetchPersonnelBreakdown, 10000); // Every 10 seconds
    const statsInterval = setInterval(fetchTodayStats, 30000); // Every 30 seconds
    const totalOnSiteInterval = setInterval(fetchTotalOnSite, 10000); // Refresh total on site also

    return () => {
      clearInterval(activityInterval);
      clearInterval(personnelInterval);
      clearInterval(statsInterval);
      clearInterval(totalOnSiteInterval);
    };
  }, [fetchActivities, fetchPersonnelBreakdown, fetchTotalOnSite, fetchTodayStats]);

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
    return new Blob([u8arr], { type: mime });
  };

  // Handler for capturing an image for Entry recognition
  const handleCaptureEntry = async (imageData: string) => {
    setIsEntryCapturing(true);
    setRecognitionMessage(null);
    setRecognitionError(null);
    try {
      if (!imageData) {
        throw new Error("No image data provided for entry capture.");
      }
      const blob = dataURLtoBlob(imageData);
      const formData = new FormData();
      formData.append("image", blob, "entry_capture.jpg"); // Changed 'file' to 'image'
      formData.append("kiosk_type", "entry"); // Add kiosk_type

      const response = await fetch(`${BACKEND_URL}/recognize_face`, { // Unified endpoint
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP error! status: ${response.status}`);
      }

      setRecognitionMessage(data.message || "Entry recognition successful!");
      fetchPersonnelBreakdown();
      fetchActivities();
      fetchTodayStats();

    } catch (err: any) {
      console.error("ERROR: during entry recognition:", err);
      setRecognitionError(`Entry recognition failed: ${err.message}. Please try again.`);
    } finally {
      setIsEntryCapturing(false);
    }
  };

  // Handler for capturing an image for Exit recognition
  const handleCaptureExit = async (imageData: string) => {
    setIsExitCapturing(true);
    setRecognitionMessage(null);
    setRecognitionError(null);
    try {
      if (!imageData) {
        throw new Error("No image data provided for exit capture.");
      }
      const blob = dataURLtoBlob(imageData);
      const formData = new FormData();
      formData.append("image", blob, "exit_capture.jpg"); // Changed 'file' to 'image'
      formData.append("kiosk_type", "exit"); // Add kiosk_type

      const response = await fetch(`${BACKEND_URL}/recognize_face`, { // Unified endpoint
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP error! status: ${response.status}`);
      }

      setRecognitionMessage(data.message || "Exit recognition successful!");
      fetchPersonnelBreakdown();
      fetchActivities();
      fetchTodayStats();

    } catch (err: any) {
      console.error("ERROR: during exit recognition:", err);
      setRecognitionError(`Exit recognition failed: ${err.message}. Please try again.`);
    } finally {
      setIsExitCapturing(false);
    }
  };

  return (
    <div className="min-h-full bg-[#101921]">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        {/* Header - This existing header content will now appear below TopButtons */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-white bg-clip-text text-transparent mb-2">
            Campus Access Monitor
          </h1>
          <p className="text-muted-foreground">
            Monitor real-time access events and campus statistics
          </p>
        </div>

        {/* Global Error Display (for initial data fetches) */}
        {fetchError && (
          <div className="bg-red-500 text-white p-3 rounded-md mb-4">
            <p className="font-semibold">Data Loading Error:</p>
            <p>{fetchError}</p>
            <p className="text-sm mt-1">Please ensure your FastAPI backend server is running correctly on {BACKEND_URL}.</p>
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
            <Card className="bg-[#1F2733] border-[#424953]  p-6">
              <h3 className="font-semibold mb-4 flex items-center space-x-2">
                <Users className="h-5 w-5 text-white" />
                <span>Personnel On Site</span>
              </h3>

              <div className="text-center mb-4">
                {loadingPersonnel ? (
                  <p className="text-xl text-muted-foreground animate-pulse">Loading...</p>
                ) : (
                  <p className="text-3xl font-bold text-[#36D399]">{totalOnSite}</p>
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
            <Card className="bg-[#1F2733] border-[#424953]  p-6">
              <h3 className="font-semibold mb-4 flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-white" />
                <span>Today's Activity</span>
              </h3>

              <div className="space-y-4">
                <div className="text-center">
                  {loadingTodayStats ? (
                    <p className="text-xl text-muted-foreground animate-pulse">Loading...</p>
                  ) : (
                    <p className="text-2xl font-bold text-[#36D399]">{todayStats.totalEntries}</p>
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
