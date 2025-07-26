// dashboard.tsx
import { useState, useEffect, useCallback } from "react";
import { Users, Shield, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CameraFeed } from "@/components/camera/CameraFeed";
import { ActivityFeed } from "@/components/activity/ActivityFeed";
import { useToast } from "@/hooks/use-toast";
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Use environment variable for the Node.js backend URL
const NODE_BACKEND_URL = import.meta.env.VITE_NODE_BACKEND_URL;

// API Endpoints through the Node.js proxy
const ACTIVITIES_API_URL = `${NODE_BACKEND_URL}/api/activities`;
const ONSITE_PERSONNEL_API_URL = `${NODE_BACKEND_URL}/api/recognition/on_site_personnel`; // Proxied
const SCANNER_STATUS_API_URL = `${NODE_BACKEND_URL}/api/recognition/scanner_status`; // Proxied
const RECOGNITION_API_HEALTH_URL = `${NODE_BACKEND_URL}/api/recognition/health`; // Proxied
const REGISTRATION_API_HEALTH_URL = `${NODE_BACKEND_URL}/api/registration/health`; // Proxied
const ENTRY_RECOGNITION_URL = `${NODE_BACKEND_URL}/api/recognition/enter_site_recognition`; // Proxied
const EXIT_RECOGNITION_URL = `${NODE_BACKEND_URL}/api/recognition/exit_site_recognition`; // Proxied


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

// Interface for On-Site Personnel response from FastAPI (proxied through Node.js)
interface UserResponse {
  id_number: string;
  name: string;
  role: string;
  on_site: boolean;
}

// Interface for Scanner Status response from FastAPI (proxied through Node.js)
interface ScannerStatusResponse {
  entry_scanner: "active" | "inactive";
  exit_scanner: "active" | "inactive";
}

// Interface for Health Status response from FastAPI (Recognition API, proxied through Node.js)
interface RecognitionHealthResponse {
  status: string; // e.g., "healthy", "unhealthy"
  database_accessible: boolean; // Specific to the recognition API's DB connection
}

// Interface for Registration API Health Response (proxied through Node.js)
interface GenericHealthResponse {
  status: string; // e.g., "healthy", "online", "offline"
}

export default function Dashboard() {
  const { toast } = useToast();
  const [onSiteCount, setOnSiteCount] = useState(0);
  const [onSiteBreakdown, setOnSiteBreakdown] = useState({
    students: 0,
    professors: 0,
    guards: 0,
    maintenance: 0,
  });
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [systemStatus, setSystemStatus] = useState({
    recognitionApi: "checking", // Status of FastAPI Recognition API
    database: "checking", // Status of Database (via Recognition API)
    activityApi: "checking", // Status of Node.js Activity API
    registrationApi: "checking", // Status of Registration API
  });
  const [isEntryKioskLoading, setIsEntryKioskLoading] = useState(false);
  const [isExitKioskLoading, setIsExitKioskLoading] = useState(false);

  // --- API Call Functions ---

  // Function to fetch and update scanner status (kept for internal logic if needed)
  const fetchScannerStatus = useCallback(async () => {
    try {
      const response = await fetch(SCANNER_STATUS_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const data: ScannerStatusResponse = await response.json();
    } catch (error) {
      console.error("Error fetching scanner status:", error);
    }
  }, []);

  // Function to fetch on-site personnel data
  const fetchOnSitePersonnel = useCallback(async () => {
    try {
      const response = await fetch(ONSITE_PERSONNEL_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: UserResponse[] = await response.json();
      setOnSiteCount(data.length);

      const breakdown = {
        students: 0,
        professors: 0,
        guards: 0,
        maintenance: 0,
      };
      data.forEach(user => {
        if (user.role in breakdown) {
          breakdown[user.role as keyof typeof breakdown]++;
        }
      });
      setOnSiteBreakdown(breakdown);
    } catch (error) {
      console.error("Error fetching on-site personnel:", error);
      setOnSiteCount(0);
      setOnSiteBreakdown({ students: 0, professors: 0, guards: 0, maintenance: 0 });
    }
  }, []);

  // Function to fetch recent activities from Node.js backend
  const fetchActivityApiStatusAndActivities = useCallback(async () => {
    try {
      const response = await fetch(ACTIVITIES_API_URL);
      if (!response.ok) {
        setSystemStatus(prev => ({ ...prev, activityApi: "offline" }));
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: any[] = await response.json();
      setSystemStatus(prev => ({ ...prev, activityApi: "healthy" })); // Set to healthy if accessible

      const transformedEntries: ActivityEntry[] = data.map((log_entry: any) => {
        const defaultId = `log-${Date.now()}-${Math.random().toFixed(4)}`;
        const defaultTimestamp = new Date().toISOString();

        const userId = String(log_entry.user_id || 'unknown_id');
        const userName = String(log_entry.userName || 'Unknown User');

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

      const latestEntries: ActivityEntry[] = [];
      const latestExits: ActivityEntry[] = [];

      for (const activity of sortedActivities) {
        if (activity.action === 'entry' && latestEntries.length < 3) {
          latestEntries.push(activity);
        } else if (activity.action === 'exit' && latestExits.length < 3) {
          latestExits.push(activity);
        }
        if (latestEntries.length === 3 && latestExits.length === 3) {
          break;
        }
      }

      const combinedActivities = [...latestEntries, ...latestExits].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

      setActivities(combinedActivities);

    } catch (error) {
      console.error("Error fetching recent activities:", error);
      setActivities([]);
    }
  }, []);

  // Function to fetch Recognition API and Database status (proxied through Node.js)
  const fetchRecognitionAndDatabaseStatus = useCallback(async () => {
    try {
      const response = await fetch(RECOGNITION_API_HEALTH_URL);
      if (!response.ok) {
        setSystemStatus(prev => ({ ...prev, recognitionApi: "offline", database: "offline" }));
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: RecognitionHealthResponse = await response.json();
      setSystemStatus(prev => ({
        ...prev,
        recognitionApi: data.status === "healthy" ? "healthy" : "offline",
        database: data.database_accessible ? "healthy" : "offline",
      }));
    } catch (error) {
      console.error("Error fetching recognition/database status:", error);
      setSystemStatus(prev => ({
        ...prev,
        recognitionApi: "offline",
        database: "offline",
      }));
    }
  }, []);

  // Function to fetch Registration API status (proxied through Node.js)
  const fetchRegistrationApiStatus = useCallback(async () => {
    try {
      const response = await fetch(REGISTRATION_API_HEALTH_URL);
      if (!response.ok) {
        setSystemStatus(prev => ({ ...prev, registrationApi: "offline" }));
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: GenericHealthResponse = await response.json();
      setSystemStatus(prev => ({
        ...prev,
        registrationApi: data.status === "healthy" || data.status === "online" ? "healthy" : "offline",
      }));
    } catch (error) {
      console.error("Error fetching registration API status:", error);
      setSystemStatus(prev => ({
        ...prev,
        registrationApi: "offline",
      }));
    }
  }, []);

  // --- useEffect Hooks for Data Fetching ---
  useEffect(() => {
    fetchScannerStatus(); // Still fetch for potential internal use
    fetchOnSitePersonnel();
    fetchActivityApiStatusAndActivities();
    fetchRecognitionAndDatabaseStatus();
    fetchRegistrationApiStatus();

    const scannerStatusInterval = setInterval(fetchScannerStatus, 5000);
    const onSitePersonnelInterval = setInterval(fetchOnSitePersonnel, 5000);
    const activitiesInterval = setInterval(fetchActivityApiStatusAndActivities, 3000);
    const recognitionDbStatusInterval = setInterval(fetchRecognitionAndDatabaseStatus, 10000);
    const registrationApiStatusInterval = setInterval(fetchRegistrationApiStatus, 10000);

    return () => {
      clearInterval(scannerStatusInterval);
      clearInterval(onSitePersonnelInterval);
      clearInterval(activitiesInterval);
      clearInterval(recognitionDbStatusInterval);
      clearInterval(registrationApiStatusInterval);
    };
  }, [fetchScannerStatus, fetchOnSitePersonnel, fetchActivityApiStatusAndActivities, fetchRecognitionAndDatabaseStatus, fetchRegistrationApiStatus]);


  // handleEntryCapture and handleExitCapture (proxied through Node.js)
  const handleEntryCapture = useCallback(async (imageData: string) => {
    setIsEntryKioskLoading(true);
    toast({
      title: "Entry Kiosk Capture",
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
      formData.append('file', blob, 'kiosk_entry_capture.png');

      const response = await fetch(ENTRY_RECOGNITION_URL, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (response.ok) {
        toast({
          title: "Entry Recognition Success",
          description: result.message || `User ${result.user?.name} recognized for entry.`,
          variant: "default",
        });
      } else {
        throw new Error(result.detail || result.message || "Recognition failed.");
      }
    } catch (error: any) {
      toast({
        title: "Entry Recognition Failed",
        description: error.message || "An error occurred during recognition.",
        variant: "destructive",
      });
    } finally {
      setIsEntryKioskLoading(false);
      fetchActivityApiStatusAndActivities();
      fetchOnSitePersonnel();
    }
  }, [toast, fetchActivityApiStatusAndActivities, fetchOnSitePersonnel]);

  const handleExitCapture = useCallback(async (imageData: string) => {
    setIsExitKioskLoading(true);
    toast({
      title: "Exit Kiosk Capture",
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
      formData.append('file', blob, 'kiosk_exit_capture.png');

      const response = await fetch(EXIT_RECOGNITION_URL, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        toast({
          title: "Exit Recognition Success",
          description: result.message || `User ${result.user?.name} recognized for exit.`,
          variant: "default",
        });
      } else {
        throw new Error(result.detail || result.message || "Recognition failed.");
      }
    } catch (error: any) {
      toast({
        title: "Exit Recognition Failed",
        description: error.message || "An error occurred during recognition.",
        variant: "destructive",
      });
    } finally {
      setIsExitKioskLoading(false);
      fetchActivityApiStatusAndActivities();
      fetchOnSitePersonnel();
    }
  }, [toast, fetchActivityApiStatusAndActivities, fetchOnSitePersonnel]);

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
            onCapture={handleEntryCapture}
            loading={isEntryKioskLoading}
          />

          <CameraFeed
            title="EXIT"
            onCapture={handleExitCapture}
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