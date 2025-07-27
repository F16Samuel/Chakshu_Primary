import { useState, useEffect, useCallback } from "react";
import { Upload, User, LogIn, LogOut, Clock, CheckCircle, AlertCircle, ArrowRight, ArrowLeft, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FileUpload } from "@/components/upload/FileUpload";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Use environment variable for the unified FastAPI backend URL
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

// Define the API URLs directly to the FastAPI backend
const RECOGNITION_API_URL = `${BACKEND_URL}/recognize_face`; // Unified recognition endpoint
const MANUAL_ENTRY_API_URL = `${BACKEND_URL}/users`; // Base for manual entry/exit
const ACTIVITIES_API_URL = `${BACKEND_URL}/dashboard/activities`; // Unified activities endpoint


// --- UPDATED INTERFACE (Remains the same as last time, it was correct for ActivityFeed) ---
interface ManualEntry {
  id: string;
  userId: string;
  userName: string;
  action: "entry" | "exit" | "entry_attempt_already_on" | "exit_attempt_already_off" | string;
  timestamp: string;
  method: "scanner" | "kiosk_capture" | string;
  confidence?: number;
  details?: string;
  status?: "success" | "failed" | "processing"; // Keep status for manual page specific rendering
}

interface RecognitionResponse {
  message: string;
  status: string; // "recognized", "unknown", "cooldown", "error"
  user_id?: string;
  name?: string; // Changed from user_name to name to match FastAPI response
  role?: string; // Changed from user_role to role to match FastAPI response
  confidence?: number; // FastAPI might return this as a float (e.g., 0.92)
}

export default function Manual() {
  const { toast } = useToast();
  const [selectedPhoto, setSelectedPhoto] = useState<File[]>([]);
  const [selectedAction, setSelectedAction] = useState<"entry" | "exit" | "">("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ManualEntry | null>(null);
  const [recentEntries, setRecentEntries] = useState<ManualEntry[]>([]);
  const [isActivitiesLoading, setIsActivitiesLoading] = useState(true);
  const [allUsers, setAllUsers] = useState<any[]>([]); // To fetch user IDs for manual entry/exit
  const [selectedUserId, setSelectedUserId] = useState<string>(""); // For manual entry/exit by ID

  // Function to fetch all users for the dropdown
  const fetchAllUsers = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/users`); // Unified endpoint for listing users
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setAllUsers(data.users);
    } catch (error) {
      console.error("Error fetching all users:", error);
      toast({
        title: "Error",
        description: "Failed to load user list for manual operations.",
        variant: "destructive",
      });
    }
  }, [toast]);

  // Function to fetch recent activities
  const fetchRecentActivities = useCallback(async () => {
    setIsActivitiesLoading(true);
    try {
      const response = await fetch(ACTIVITIES_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: any[] = await response.json();

      const transformedEntries: ManualEntry[] = data.map((log_entry: any) => {
        const defaultId = `log-${Date.now()}-${Math.random().toFixed(4)}`;
        const defaultTimestamp = new Date().toISOString();

        const userId = String(log_entry.user_id || 'unknown_id');
        const userName = String(log_entry.user_name || 'Unknown User'); // Matches FastAPI

        const backendAction = String(log_entry.action || 'unknown').toLowerCase();
        let entryAction: ManualEntry['action'];
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
                entryAction = 'unknown_action';
        }

        const entryStatus: ManualEntry['status'] =
            (userName !== 'Unknown User' && (entryAction === 'entry' || entryAction === 'exit'))
            ? 'success'
            : 'failed';

        const entryMethod: ManualEntry['method'] =
            log_entry.method?.toLowerCase() === 'scanner' ? 'scanner' :
            log_entry.method?.toLowerCase() === 'kiosk_capture' ? 'kiosk_capture' :
            'unknown_method';

        const entryConfidence = typeof log_entry.confidence === 'number' ? log_entry.confidence : undefined;
        const entryDetails = log_entry.details || log_entry.message || log_entry.error_message || '';

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
      }).filter(entry => entry.action !== 'unknown_action');

      setRecentEntries(transformedEntries.slice(0, 10));
    } catch (error) {
      console.error("Error fetching recent activities:", error);
      toast({
        title: "Failed to Load Activities",
        description: "Could not fetch recent activity logs. Check browser console for details.",
        variant: "destructive",
      });
      setRecentEntries([]);
    } finally {
      setIsActivitiesLoading(false);
    }
  }, [toast]);

  // Fetch activities and users when component mounts
  useEffect(() => {
    fetchRecentActivities();
    fetchAllUsers();
    const activitiesInterval = setInterval(fetchRecentActivities, 3000);
    const usersInterval = setInterval(fetchAllUsers, 10000); // Refresh user list
    return () => {
      clearInterval(activitiesInterval);
      clearInterval(usersInterval);
    };
  }, [fetchRecentActivities, fetchAllUsers]);

  // --- handleProcess Function for Photo Recognition ---
  const handleProcess = async () => {
    if (!selectedPhoto.length || !selectedAction) {
      toast({
        title: "Incomplete Form",
        description: "Please select a photo and action",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setResult(null);
    setRecentEntries(prev => [{
      id: Date.now().toString() + '-processing',
      timestamp: new Date().toISOString(),
      action: selectedAction,
      status: "processing",
      userId: '...', userName: 'Processing...', method: 'kiosk_capture'
    }, ...prev]);

    try {
      const formData = new FormData();
      formData.append("image", selectedPhoto[0]); // Changed 'file' to 'image' for FastAPI
      formData.append("kiosk_type", selectedAction); // Add kiosk_type for FastAPI

      console.log(`Sending recognition request to: ${RECOGNITION_API_URL}`);

      const response = await fetch(RECOGNITION_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw errorData;
      }

      const apiResult: RecognitionResponse = await response.json();
      console.log("API Recognition Response:", apiResult);

      // Map RecognitionResponse to ManualEntry (ActivityEntry compatible)
      const newEntry: ManualEntry = {
        id: apiResult.user_id || Date.now().toString(),
        timestamp: new Date().toISOString(),
        action: selectedAction,
        status: apiResult.status === "recognized" ? "success" : "failed", // Use backend status
        confidence: apiResult.confidence,
        userId: apiResult.user_id || 'unknown_id',
        userName: apiResult.name || 'Unknown User', // Use 'name' from FastAPI
        method: 'kiosk_capture',
        details: apiResult.status === "recognized" ? undefined : (apiResult.message || 'Recognition failed.')
      };

      setResult(newEntry);

      setRecentEntries(prev => {
        const updated = prev.map(entry =>
          entry.id.endsWith('-processing') && entry.action === selectedAction
            ? { ...newEntry, id: newEntry.id || Date.now().toString() }
            : entry
        );
        if (!updated.some(entry => entry.id === newEntry.id)) {
            updated.unshift(newEntry);
        }
        return updated.slice(0, 10);
      });

      if (newEntry.status === "success") {
        toast({
          title: "Recognition Successful",
          description: `${newEntry.action} processed successfully for ${newEntry.userName}.`,
        });
      } else {
        toast({
          title: "Recognition Failed",
          description: newEntry.details || apiResult.message || "Could not recognize the person in the photo.",
          variant: "destructive",
        });
      }

    } catch (error) {
      console.error("Error during recognition process:", error);
      setErrorStateDuringProcessing(selectedAction, error);
    } finally {
      setIsProcessing(false);
    }
  };

  // --- handleManualById Function for Manual Entry/Exit by ID ---
  const handleManualById = async (action: "entry" | "exit") => {
    if (!selectedUserId) {
      toast({
        title: "Missing User ID",
        description: "Please select a user ID for manual operation.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setResult(null); // Clear previous result
    setRecentEntries(prev => [{
      id: Date.now().toString() + '-processing-id',
      timestamp: new Date().toISOString(),
      action: action,
      status: "processing",
      userId: selectedUserId, userName: 'Processing...', method: 'manual'
    }, ...prev]);

    try {
      const endpoint = `${MANUAL_ENTRY_API_URL}/${selectedUserId}/manual_${action}`;
      console.log(`Sending manual ${action} request to: ${endpoint}`);

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const apiResult = await response.json();

      if (!response.ok) {
        throw apiResult; // FastAPI errors have 'detail'
      }

      const newEntry: ManualEntry = {
        id: selectedUserId + '-' + Date.now().toString(),
        timestamp: new Date().toISOString(),
        action: action,
        status: "success",
        userId: selectedUserId,
        userName: apiResult.message.includes("logged for") ? apiResult.message.split("logged for ")[1].replace(".", "") : "User", // Extract name from message
        method: 'manual',
        details: apiResult.message
      };
      setResult(newEntry);
      toast({
        title: "Manual Operation Successful",
        description: apiResult.message,
      });

      setRecentEntries(prev => {
        const updated = prev.map(entry =>
          entry.id.endsWith('-processing-id') && entry.userId === selectedUserId && entry.action === action
            ? { ...newEntry, id: newEntry.id || Date.now().toString() }
            : entry
        );
        if (!updated.some(entry => entry.id === newEntry.id)) {
            updated.unshift(newEntry);
        }
        return updated.slice(0, 10);
      });

    } catch (error) {
      console.error("Error during manual operation:", error);
      let errorMessage = "An unknown error occurred.";
      if (typeof error === 'object' && error !== null && 'detail' in error) {
          errorMessage = error.detail;
      } else if (error instanceof Error) {
          errorMessage = error.message;
      } else if (typeof error === 'string') {
          errorMessage = error;
      }
      const errorEntry: ManualEntry = {
        id: selectedUserId + '-' + Date.now().toString() + '-error',
        timestamp: new Date().toISOString(),
        action: action,
        status: "failed",
        userId: selectedUserId,
        userName: 'Unknown User',
        method: 'manual',
        details: errorMessage,
      };
      setResult(errorEntry);
      setRecentEntries(prev => {
        const updated = prev.map(entry =>
          entry.id.endsWith('-processing-id') && entry.userId === selectedUserId && entry.action === action
            ? { ...errorEntry, id: errorEntry.id || Date.now().toString() }
            : entry
        );
        if (!updated.some(entry => entry.id === errorEntry.id)) {
          updated.unshift(errorEntry);
        }
        return updated.slice(0, 10);
      });
      toast({
        title: "Manual Operation Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
      setSelectedUserId(""); // Clear selected user ID after processing
    }
  };


  const setErrorStateDuringProcessing = useCallback((action: "entry" | "exit" | "", error: any) => {
    let errorMessage = "An unknown error occurred.";
    if (error instanceof Error) {
        errorMessage = error.message;
    } else if (typeof error === 'object' && error !== null && 'detail' in error) {
        if (Array.isArray(error.detail) && error.detail.length > 0) {
            errorMessage = error.detail.map((err: any) => {
                const loc = Array.isArray(err.loc) ? err.loc.join('.') : 'unknown_location';
                return `${loc} - ${err.msg}`;
            }).join('; ');
        } else if (typeof error.detail === 'string') {
            errorMessage = error.detail;
        }
    } else if (typeof error === 'string') {
        errorMessage = error;
    }

    const errorEntry: ManualEntry = {
      id: Date.now().toString() + '-error',
      timestamp: new Date().toISOString(),
      action: action,
      status: "failed",
      userId: 'unknown_id', // Default for failed manual entry
      userName: 'Unknown User', // Default for failed manual entry
      method: 'kiosk_capture', // Failed manual entry method
      details: errorMessage,
    };
    setResult(errorEntry);
    setRecentEntries(prev => {
      const updated = prev.map(entry =>
        entry.id.endsWith('-processing') && entry.action === action
          ? { ...errorEntry, id: errorEntry.id || Date.now().toString() }
          : entry
      );
      if (!updated.some(entry => entry.id === errorEntry.id)) {
        updated.unshift(errorEntry);
      }
      return updated.slice(0, 10);
    });

    toast({
      title: "Processing Error",
      description: errorMessage,
      variant: "destructive",
    });
  }, [toast]);

  const handlePhotoSelect = (files: File[]) => {
    setSelectedPhoto(files);
    setResult(null);
  };

  const handleReset = () => {
    setSelectedPhoto([]);
    setSelectedAction("");
    setResult(null);
    setSelectedUserId("");
  };

  const formatTime = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch (e) {
      console.error("Invalid timestamp:", timestamp, e);
      return "Invalid Time";
    }
  };

  const formatDate = (timestamp: string) => {
    try {
        return new Date(timestamp).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
        });
    } catch (e) {
        console.error("Invalid timestamp for date:", timestamp, e);
        return "Invalid Date";
    }
  };

  // --- Helper functions from ActivityFeed.tsx for consistent display ---
  const getActionIcon = (action: ManualEntry['action']) => {
    switch (action) {
      case "entry":
        return <ArrowRight className="h-4 w-4 text-[#36D399]" />;
      case "exit":
        return <ArrowLeft className="h-4 w-4 text-red-600" />;
      case "entry_attempt_already_on":
      case "exit_attempt_already_off":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <User className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getActionColor = (action: ManualEntry['action']) => {
    switch (action) {
      case "entry":
        return "text-[#36D399] bg-[#1F2733] border-[#424953]";
      case "exit":
        return "text-red-500 bg-[#1F2733] border-[#424953]";
      case "entry_attempt_already_on":
      case "exit_attempt_already_off":
        return "text-yellow-500 border-yellow-500/30 bg-yellow-500/10";
      default:
        return "text-muted-foreground border-muted/30 bg-muted/10";
    }
  };

  const getActionLabel = (action: ManualEntry['action']) => {
    switch (action) {
        case "entry": return "Entry";
        case "exit": return "Exit";
        case "entry_attempt_already_on": return "Entry Attempt (On-Site)";
        case "exit_attempt_already_off": return "Exit Attempt (Off-Site)";
        default: return "Unknown Action";
    }
  };
  // --- END NEW HELPER FUNCTIONS ---


  // --- KEEP THESE MANUAL PAGE SPECIFIC HELPERS (getStatusIcon/Badge are for the main result, not recent entries) ---
  const getStatusIcon = (status: ManualEntry['status']) => {
    switch (status) {
      case "success":
        return <CheckCircle className="h-4 w-4 text-[#36D399]" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-red" />;
      case "processing":
        return <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: ManualEntry['status']) => {
    switch (status) {
      case "success":
        return <Badge variant="default">Success</Badge>;
      case "failed":
        return <Badge variant="destructive">Failed</Badge>;
      case "processing":
        return <Badge variant="secondary">Processing</Badge>;
      default:
        return null;
    }
  };
  // --- END MANUAL PAGE SPECIFIC HELPERS ---

  return (
    <div className="min-h-full bg-[#101921]">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        {/* Header - This existing header content will now appear below TopButtons */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-white bg-clip-text text-transparent mb-2">
            Manual Recognition
          </h1>
          <p className="text-muted-foreground">
            Process entry/exit manually using photo upload or user ID
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Processing Panel */}
          <div className="space-y-6">
            {/* Upload Section */}
            <Card className="bg-[#1F2733] border-[#424953] p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <Upload className="h-5 w-5 text-white" />
                <span>Photo Upload Recognition</span>
              </h2>

              <FileUpload
                accept="image/*"
                label="Select Face Photo"
                onFileSelect={handlePhotoSelect}
                maxSize={10}
              />

              <div className="space-y-4 mt-6">
                <Select
                  value={selectedAction}
                  onValueChange={(value) => setSelectedAction(value as "entry" | "exit" | "")}
                >
                  <SelectTrigger className="bg-[#101921] border-[#1F2733]">
                    <SelectValue placeholder="Select action" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="entry">
                      <div className="flex items-center space-x-2">
                        <LogIn className="h-4 w-4 text-[#34CD95]" />
                        <span>Entry (Photo)</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="exit">
                      <div className="flex items-center space-x-2">
                        <LogOut className="h-4 w-4 text-[red]" />
                        <span>Exit (Photo)</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>

                <Button
                  variant="cyber"
                  onClick={handleProcess}
                  disabled={!selectedPhoto.length || !selectedAction || isProcessing}
                  className="w-full"
                >
                  {isProcessing ? "Processing..." : "Process Photo Recognition"}
                </Button>
              </div>
            </Card>

            {/* Manual Entry/Exit by ID */}
            <Card className="bg-[#1F2733] border-[#424953] p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <User className="h-5 w-5 text-white" />
                <span>Manual Entry/Exit by ID</span>
              </h2>
              <div className="space-y-4">
                <Select
                  value={selectedUserId}
                  onValueChange={setSelectedUserId}
                  disabled={isProcessing}
                >
                  <SelectTrigger className="bg-[#101921] border-[#1F2733]">
                    <SelectValue placeholder="Select User ID" />
                  </SelectTrigger>
                  <SelectContent>
                    {allUsers.map((user) => (
                      <SelectItem key={user.id_number} value={user.id_number}>
                        {user.name} ({user.id_number})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex space-x-4">
                  <Button
                    variant="cyber"
                    onClick={() => handleManualById("entry")}
                    disabled={!selectedUserId || isProcessing}
                    className="flex-1"
                  >
                    <LogIn className="h-4 w-4 mr-2" />
                    Manual Entry
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => handleManualById("exit")}
                    disabled={!selectedUserId || isProcessing}
                    className="flex-1"
                  >
                    <LogOut className="h-4 w-4 mr-2" />
                    Manual Exit
                  </Button>
                </div>
              </div>
            </Card>

            {/* Results Section */}
            {(result || isProcessing) && (
              <Card className="bg-[#1F2733] border-[#1F2733] p-6">
                <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-white" />
                  <span>Results</span>
                </h2>

                {isProcessing ? (
                  <div className="text-center py-8">
                    <div className="h-12 w-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-white font-medium">Processing recognition...</p>
                    <p className="text-sm text-muted-foreground mt-1">This may take a few seconds</p>
                  </div>
                ) : result && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Status</span>
                      {getStatusBadge(result.status)}
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="font-medium">Action</span>
                      <Badge variant="secondary" className="capitalize">
                        {getActionLabel(result.action)}
                      </Badge>
                    </div>

                    {result.confidence !== undefined && (
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Confidence</span>
                        <span className="text-primary font-mono">
                          {/* Format confidence to 2 decimal places */}
                          {result.confidence.toFixed(2)}%
                        </span>
                      </div>
                    )}

                    {result.userName !== 'Unknown User' && (
                      <div className="border-t border-border pt-4">
                        <h4 className="font-medium mb-2">Matched User</h4>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">Name</span>
                            <span className="text-sm font-medium">{result.userName}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm">ID</span>
                            <span className="text-sm font-mono">{result.userId}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    {result.status === "failed" && result.details && (
                      <p className="text-sm text-destructive-foreground mt-2">Reason: {result.details}</p>
                    )}

                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                      <span>Processed at</span>
                      <span className="font-mono">{formatTime(result.timestamp)}</span>
                    </div>
                  </div>
                )}
                <div className="flex justify-center mt-4">
                    <Button
                        variant="outline"
                        onClick={handleReset}
                        disabled={isProcessing}
                    >
                        Reset Form
                    </Button>
                </div>
              </Card>
            )}
          </div>

          {/* Recent Entries */}
          <Card className="bg-[#101921] border-[#1F2733]">
            <div className="p-6 border-b border-[#1F2733]">
              <h2 className="text-xl font-semibold flex items-center space-x-2">
                <Clock className="h-5 w-5 text-white" />
                <span>Recent Manual Entries</span>
                {isActivitiesLoading && <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin ml-2" />}
              </h2>
            </div>

            <div className="max-h-96 overflow-y-auto">
              {isActivitiesLoading ? (
                <div className="p-8 text-center">
                  <div className="h-12 w-12 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                  <p className="text-muted-foreground">Loading recent activities...</p>
                </div>
              ) : recentEntries.length === 0 ? (
                <div className="p-8 text-center">
                  <User className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                  <p className="text-muted-foreground">No recent manual entries yet</p>
                </div>
              ) : (
                <div className="p-4 space-y-3">
                  {recentEntries.map((entry, index) => {
                    const isToday = new Date(entry.timestamp).toDateString() === new Date().toDateString();
                    const confidenceDisplay = entry.confidence !== undefined && entry.confidence !== null
                        ? `${entry.confidence.toFixed(2)}% confidence`
                        : 'N/A';

                    return (
                      <div
                        key={entry.id}
                        className={cn(
                          "group relative p-3 rounded-lg transition-all duration-200 hover:bg-[#52525B]",
                          index === 0 && ""
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            {/* Action Icon */}
                            <div className={cn(
                              "flex items-center justify-center w-8 h-8 rounded-full border",
                              getActionColor(entry.action)
                            )}>
                              {getActionIcon(entry.action)}
                            </div>

                            {/* User Info */}
                            <div className="flex-1">
                              <div className="flex items-center space-x-2">
                                <span className="font-medium text-sm">
                                  {entry.userName}
                                </span>
                                <Badge
                                  variant="outline"
                                  className={cn(
                                    "text-xs capitalize",
                                    getActionColor(entry.action)
                                  )}
                                >
                                  {getActionLabel(entry.action)}
                                </Badge>
                              </div>

                              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                                <span>ID: {entry.userId}</span>
                                {entry.method === "kiosk_capture" && (
                                  <Badge variant="secondary" className="text-xs">
                                    Capture
                                  </Badge>
                                )}
                                {entry.confidence !== undefined && entry.confidence !== null && (
                                  <span className="text-[#9A9A9B]">
                                    {confidenceDisplay}
                                  </span>
                                )}
                              </div>
                              {entry.status === "failed" && entry.details && (
                                <p className="text-xs text-destructive-foreground mt-1">Reason: {entry.details}</p>
                              )}
                            </div>
                          </div>

                          {/* Timestamp */}
                          <div className="text-right text-xs text-muted-foreground">
                            <div className="font-mono">
                              {formatTime(entry.timestamp)}
                            </div>
                            {!isToday && (
                              <div className="text-xs">
                                {formatDate(entry.timestamp)}
                              </div>
                            )}
                          </div>
                        </div>

                        {/* New entry indicator */}
                        {index === 0 && (
                          <div className="" />
                        )}
                      </div>
                    );
                  })}
                </div>
            )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
