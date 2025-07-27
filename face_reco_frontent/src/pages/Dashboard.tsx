// src/pages/Dashboard.tsx
import { useState, useEffect, useCallback } from "react";
import { Users, Shield, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CameraFeed } from "@/components/camera/CameraFeed";
import { ActivityFeed } from "@/components/activity/ActivityFeed";
import { useToast } from "@/hooks/use-toast";
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Use environment variable for the unified FastAPI backend URL
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

// API Endpoints directly to the FastAPI backend
const ACTIVITIES_API_URL = `${BACKEND_URL}/dashboard/activities`;
const ONSITE_PERSONNEL_API_URL = `${BACKEND_URL}/dashboard/personnel-breakdown`; // This now gives breakdown, not just on-site count
const TOTAL_ONSITE_API_URL = `${BACKEND_URL}/dashboard/total-on-site`; // New endpoint for total on-site
const RECOGNITION_API_HEALTH_URL = `${BACKEND_URL}/health`; // Unified health check
const REGISTRATION_API_HEALTH_URL = `${BACKEND_URL}/health`; // Unified health check
const ENTRY_RECOGNITION_URL = `${BACKEND_URL}/recognize_face`; // Unified recognition endpoint
const EXIT_RECOGNITION_URL = `${BACKEND_URL}/recognize_face`; // Unified recognition endpoint
const TODAY_STATS_API_URL = `${BACKEND_URL}/dashboard/today-stats`; // For today's stats


// Interface for activity logs - MUST match ActivityFeed.tsx exactly
interface ActivityEntry {
  id: string;
  userId: string;
  userName: string;
  action: "entry" | "exit" | "entry_attempt_already_on" | "exit_attempt_already_off";
  timestamp: string;
  method: "scanner" | "kiosk_capture";
  confidence?: number;
  details?: string;
  status?: "success" | "failed" | "processing";
}

// Interface for Personnel Breakdown response from FastAPI
interface PersonnelBreakdownResponse {
  students: number;
  professors: number;
  guards: number;
  maintenance: number;
}

// Interface for Total On-Site response from FastAPI
interface TotalOnSiteResponse {
  totalOnSite: number;
}

// Interface for Today's Stats response from FastAPI
interface TodayStatsResponse {
  totalEntries: number;
  entries: number;
  exits: number;
  peakHour: string;
}

// Interface for Health Status response from FastAPI
interface HealthResponse {
  status: string; // e.g., "ok", "error"
  database: string; // e.g., "connected", "disconnected"
  detail?: string;
}

export default function Dashboard() {
  const { toast } = useToast();
  const [onSiteCount, setOnSiteCount] = useState(0);
  const [onSiteBreakdown, setOnSiteBreakdown] = useState<PersonnelBreakdownResponse>({
    students: 0,
    professors: 0,
    guards: 0,
    maintenance: 0,
  });
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [systemStatus, setSystemStatus] = useState({
    recognitionApi: "checking", // Status of FastAPI API
    database: "checking", // Status of Database (via FastAPI)
    activityApi: "checking", // Status of FastAPI Activity API (unified)
    registrationApi: "checking", // Status of FastAPI Registration API (unified)
  });
  const [isEntryKioskLoading, setIsEntryKioskLoading] = useState(false);
  const [isExitKioskLoading, setIsExitKioskLoading] = useState(false);
  const [todayStats, setTodayStats] = useState<TodayStatsResponse>({
    totalEntries: 0,
    entries: 0,
    exits: 0,
    peakHour: "N/A",
  });
  const [loadingTodayStats, setLoadingTodayStats] = useState(true);


  // --- API Call Functions ---

  // Function to fetch on-site personnel breakdown and total
  const fetchOnSitePersonnelAndTotal = useCallback(async () => {
    try {
      const breakdownResponse = await fetch(ONSITE_PERSONNEL_API_URL);
      if (!breakdownResponse.ok) {
        throw new Error(`HTTP error! status: ${breakdownResponse.status}`);
      }
      const breakdownData: PersonnelBreakdownResponse = await breakdownResponse.json();
      setOnSiteBreakdown(breakdownData);

      // Calculate total on-site from breakdown
      const total = breakdownData.students + breakdownData.professors + breakdownData.guards + breakdownData.maintenance;
      setOnSiteCount(total);

    } catch (error) {
      console.error("Error fetching on-site personnel breakdown:", error);
      setOnSiteCount(0);
      setOnSiteBreakdown({ students: 0, professors: 0, guards: 0, maintenance: 0 });
    }
  }, []);

  // Function to fetch recent activities
  const fetchActivities = useCallback(async () => {
    try {
      const response = await fetch(ACTIVITIES_API_URL);
      if (!response.ok) {
        setSystemStatus(prev => ({ ...prev, activityApi: "offline" }));
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: any[] = await response.json();
      setSystemStatus(prev => ({ ...prev, activityApi: "healthy" }));

      const transformedEntries: ActivityEntry[] = data.map((log_entry: any) => {
        const defaultId = `log-${Date.now()}-${Math.random().toFixed(4)}`;
        const defaultTimestamp = new Date().toISOString();

        const userId = String(log_entry.user_id || 'unknown_id');
        const userName = String(log_entry.user_name || 'Unknown User'); // Changed from 'userName' to 'user_name' to match FastAPI

        const backendAction = String(log_entry.action || 'unknown').toLowerCase();
        let entryAction: ActivityEntry['action'];
        switch (backendAction) {
          case 'entry':
            entryAction = 'entry';
            break;
          case 'exit':
            entryAction = 'exit';
            break;
          case 'entry_attempt_already_on':
            entryAction = 'entry_attempt_already_on';
            break;
          case 'exit_attempt_already_off':
            entryAction = 'exit_attempt_already_off';
            break;
          default:
            entryAction = 'entry'; // Default to a valid action type to avoid type issues
        }

        const entryMethod: ActivityEntry['method'] =
          log_entry.method?.toLowerCase() === 'scanner' ? 'scanner' :
          log_entry.method?.toLowerCase() === 'kiosk_capture' ? 'kiosk_capture' :
          'scanner';

        const entryConfidence = typeof log_entry.confidence === 'number' ? log_entry.confidence : undefined;
        const entryDetails = log_entry.details || log_entry.message || log_entry.error_message || '';
        const entryStatus: ActivityEntry['status'] = 'success';

        return {
          id: String(log_entry.id || defaultId),
          timestamp: String(log_entry.timestamp || defaultTimestamp),
          userId: userId,
          userName: userName,
          action: entryAction,
          status: entryStatus,
          method: entryMethod,
          confidence: entryConfidence,
          details: entryDetails,
        };
      }).filter(entry =>
        entry.action === 'entry' ||
        entry.action === 'exit' ||
        entry.action === 'entry_attempt_already_on' ||
        entry.action === 'exit_attempt_already_off'
      );

      const sortedActivities = transformedEntries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      // Display all recent activities, not just a filtered few
      setActivities(sortedActivities);

    } catch (error) {
      console.error("Error fetching recent activities:", error);
      setActivities([]);
    }
  }, []);

  // Function to fetch unified API health status
  const fetchApiHealthStatus = useCallback(async () => {
    try {
      const response = await fetch(RECOGNITION_API_HEALTH_URL); // Using the unified health endpoint
      if (!response.ok) {
        setSystemStatus(prev => ({ ...prev, recognitionApi: "offline", database: "offline", registrationApi: "offline" }));
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: HealthResponse = await response.json();
      setSystemStatus(prev => ({
        ...prev,
        recognitionApi: data.status === "ok" ? "healthy" : "offline",
        database: data.database === "connected" ? "healthy" : "offline",
        registrationApi: data.status === "ok" ? "healthy" : "offline", // Registration API status is now tied to main health
      }));
    } catch (error) {
      console.error("Error fetching API health status:", error);
      setSystemStatus(prev => ({
        ...prev,
        recognitionApi: "offline",
        database: "offline",
        registrationApi: "offline",
      }));
    }
  }, []);

  // Function to fetch today's statistics
  const fetchTodayStats = useCallback(async () => {
    setLoadingTodayStats(true);
    try {
      const response = await fetch(TODAY_STATS_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: TodayStatsResponse = await response.json();
      setTodayStats(data);
    } catch (err: any) {
      console.error("Error fetching today's stats:", err);
      setTodayStats({ totalEntries: 0, entries: 0, exits: 0, peakHour: "N/A" });
    } finally {
      setLoadingTodayStats(false);
    }
  }, []);

  // --- useEffect Hooks for Data Fetching ---
  useEffect(() => {
    fetchOnSitePersonnelAndTotal();
    fetchActivities();
    fetchApiHealthStatus();
    fetchTodayStats(); // Fetch initial today's stats

    const onSitePersonnelInterval = setInterval(fetchOnSitePersonnelAndTotal, 5000);
    const activitiesInterval = setInterval(fetchActivities, 3000);
    const apiHealthInterval = setInterval(fetchApiHealthStatus, 10000);
    const todayStatsInterval = setInterval(fetchTodayStats, 15000); // Refresh every 15 seconds

    return () => {
      clearInterval(onSitePersonnelInterval);
      clearInterval(activitiesInterval);
      clearInterval(apiHealthInterval);
      clearInterval(todayStatsInterval);
    };
  }, [fetchOnSitePersonnelAndTotal, fetchActivities, fetchApiHealthStatus, fetchTodayStats]);


  // handleEntryCapture and handleExitCapture (now directly to FastAPI)
  const handleCapture = useCallback(async (imageData: string, kioskType: "entry" | "exit") => {
    if (kioskType === "entry") setIsEntryKioskLoading(true);
    else setIsExitKioskLoading(true);

    toast({
      title: `${kioskType === "entry" ? "Entry" : "Exit"} Kiosk Capture`,
      description: "Image captured. Sending for recognition...",
      duration: 3000,
    });

    try {
      const byteString = atob(imageData.split(',')[1]);
      const mimeString = imageData.split(',')[0].split(':')[1].split(';')[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([ab], { type: mimeString });

      const formData = new FormData();
      formData.append('image', blob, `${kioskType}_capture.png`); // Changed 'file' to 'image' for FastAPI
      formData.append('kiosk_type', kioskType); // Add kiosk_type

      const response = await fetch(BACKEND_URL + '/recognize_face', { // Unified endpoint
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (response.ok) {
        toast({
          title: `${kioskType === "entry" ? "Entry" : "Exit"} Recognition Success`,
          description: result.message || `User ${result.name} recognized for ${kioskType}.`,
          variant: "default",
        });
      } else {
        throw new Error(result.detail || result.message || "Recognition failed.");
      }
    } catch (error: any) {
      toast({
        title: `${kioskType === "entry" ? "Entry" : "Exit"} Recognition Failed`,
        description: error.message || "An error occurred during recognition.",
        variant: "destructive",
      });
    } finally {
      if (kioskType === "entry") setIsEntryKioskLoading(false);
      else setIsExitKioskLoading(false);
      fetchActivities(); // Refresh activities after recognition attempt
      fetchOnSitePersonnelAndTotal(); // Refresh personnel count
      fetchTodayStats(); // Refresh today's stats
    }
  }, [toast, fetchActivities, fetchOnSitePersonnelAndTotal, fetchTodayStats]);

  // Helper for System Status badge color
  const getStatusVariant = (status: string) => {
    return status === "healthy" ? "default" : "destructive";
  };

  // Determine overall system health message
  const isOverallSystemHealthy =
    systemStatus.recognitionApi === "healthy" &&
    systemStatus.database === "healthy" &&
    systemStatus.activityApi === "healthy" &&
    systemStatus.registrationApi === "healthy";

  return (
    <div className="min-h-full bg-[#101921]">
      <TopButtons /> {/* Added the TopButtons component here */}
      <div className="container mx-auto px-4 py-8 space-y-8">
        {/* Header Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="bg-[#1F2733] border-[#1F2733] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground text-sm">On Site </p>
                <p className="text-3xl font-bold text-white">{onSiteCount}</p>
                <div className="text-xs text-muted-foreground mt-2">
                  <p>Students: {onSiteBreakdown.students}</p>
                  <p>Professors: {onSiteBreakdown.professors}</p>
                  <p>Guards: {onSiteBreakdown.guards}</p>
                  <p>Maintenance: {onSiteBreakdown.maintenance}</p>
                </div>
              </div>
              <Users className="h-8 w-8 text-white" />
            </div>
          </Card>

          <Card className="bg-[#1F2733] border-[#1F2733] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground text-sm">System Status</p>
                <Badge variant={getStatusVariant(isOverallSystemHealthy ? "healthy" : "offline")}>
                  {isOverallSystemHealthy ? "Online" : "Offline"}
                </Badge>
              </div>
              <Shield className="h-8 w-8 text-white" />
            </div>
          </Card>
        </div>

        {/* Camera Feeds (still for manual capture) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <CameraFeed
            title="ENTRY"
            onCapture={(imageData) => handleCapture(imageData, "entry")}
            loading={isEntryKioskLoading}
          />

          <CameraFeed
            title="EXIT"
            onCapture={(imageData) => handleCapture(imageData, "exit")}
            loading={isExitKioskLoading}
          />
        </div>

        {/* Activity Feed and System Health Check */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <ActivityFeed activities={activities} maxItems={6} />
          </div>

          <Card className="bg-[#1F2733] border-[#1F2733]/30">
            <div className="p-6">
              <h3 className="font-semibold text-lg mb-4 flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-[#36D399]" />
                <span>System Health Check</span>
              </h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Recognition API</span>
                  <Badge variant={getStatusVariant(systemStatus.recognitionApi)}>
                    {systemStatus.recognitionApi === "healthy" ? "Healthy" : "Offline"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Registration API</span>
                  <Badge variant={getStatusVariant(systemStatus.registrationApi)}>
                    {systemStatus.registrationApi === "healthy" ? "Healthy" : "Offline"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Database Access</span>
                  <Badge variant={getStatusVariant(systemStatus.database)}>
                    {systemStatus.database === "healthy" ? "Healthy" : "Offline"}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Activity API</span>
                  <Badge variant={getStatusVariant(systemStatus.activityApi)}>
                    {systemStatus.activityApi === "healthy" ? "Healthy" : "Offline"}
                  </Badge>
                </div>
              </div>

              <div className="mt-6 p-4 bg-[#1F2733]/10 rounded-lg border border-[#101921]">
                <p className="text-sm text-white font-medium flex items-center">
                  {isOverallSystemHealthy
                    ? (<><CheckCircle className="h-4 w-4 mr-2 text-[#36D399]" /> All core systems operational.</>)
                    : (<><XCircle className="h-4 w-4 mr-2 text-destructive" /> Some systems are offline or degraded.</>)
                  }
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Last checked: {new Date().toLocaleTimeString()}
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
