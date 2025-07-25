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

// Use environment variable for the Node.js backend URL
const NODE_BACKEND_URL = import.meta.env.VITE_NODE_BACKEND_URL;

// Define the API URLs via the Node.js proxy
const RECOGNITION_API_ENTER_URL = `${NODE_BACKEND_URL}/api/recognition/enter_site_recognition`; // Proxied
const RECOGNITION_API_EXIT_URL = `${NODE_BACKEND_URL}/api/recognition/exit_site_recognition`; // Proxied
const ACTIVITIES_API_URL = `${NODE_BACKEND_URL}/api/activities`; // Already correct


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
  success: boolean;
  user_id?: string;
  user_name?: string;
  user_role?: string;
  confidence?: number; // FastAPI might return this as a float (e.g., 0.92) or a percentage (92.0)
}

export default function Manual() {
  const { toast } = useToast();
  const [selectedPhoto, setSelectedPhoto] = useState<File[]>([]);
  const [selectedAction, setSelectedAction] = useState<"entry" | "exit" | "">("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<ManualEntry | null>(null);
  const [recentEntries, setRecentEntries] = useState<ManualEntry[]>([]);
  const [isActivitiesLoading, setIsActivitiesLoading] = useState(true);

  // --- Functions to fetch and process data ---

  // Function to fetch recent activities from Node.js backend
  const fetchRecentActivities = useCallback(async () => {
    setIsActivitiesLoading(true);
    try {
      const response = await fetch(ACTIVITIES_API_URL);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: any[] = await response.json(); // Raw data from your Node.js backend

      console.log("Raw activities data from Node.js:", data); // Log the raw data for debugging

      const transformedEntries: ManualEntry[] = data.map((log_entry: any) => {
        const defaultId = `log-${Date.now()}-${Math.random().toFixed(4)}`;
        const defaultTimestamp = new Date().toISOString();

        // **1. Extract User Details (Direct Access):**
        // Based on your provided JSON, these fields are top-level.
        const userId = String(log_entry.user_id || 'unknown_id');
        const userName = String(log_entry.userName || 'Unknown User'); // Backend sends 'userName' directly

        // **2. Determine Action Type:**
        const backendAction = String(log_entry.action || 'unknown').toLowerCase();
        let entryAction: ManualEntry['action'];
        switch (backendAction) {
            case 'entry':
                entryAction = 'entry';
                break;
            case 'exit':
                entryAction = 'exit';
                break;
            case 'entry_attempt_already_on': // Ensure your backend sends these if applicable
                entryAction = 'entry_attempt_already_on';
                break;
            case 'exit_attempt_already_off': // Ensure your backend sends these if applicable
                entryAction = 'exit_attempt_already_off';
                break;
            default:
                entryAction = 'unknown_action';
        }

        // **3. Determine Status (for Manual page display - inferred if not explicit in activity log):**
        // Your activity log doesn't have an explicit 'success' or 'status' field.
        // We can infer 'success' if a userName is present and action is a clear entry/exit.
        // Or you might need to add a 'status' field to your Node.js activity logs for more accuracy.
        // For now, let's assume if userName is present and it's a valid action, it's 'success'.
        const entryStatus: ManualEntry['status'] = 
            (userName !== 'Unknown User' && (entryAction === 'entry' || entryAction === 'exit')) 
            ? 'success' 
            : 'failed'; // Mark as failed if user is unknown or action is an 'attempt' or unknown


        // **4. Determine Method:**
        const entryMethod: ManualEntry['method'] = 
            log_entry.method?.toLowerCase() === 'scanner' ? 'scanner' :
            log_entry.method?.toLowerCase() === 'kiosk_capture' ? 'kiosk_capture' :
            'unknown_method';

        // **5. Other Fields:**
        // Confidence is now directly used without multiplying by 100 here.
        const entryConfidence = typeof log_entry.confidence === 'number' ? log_entry.confidence : undefined;
        // Your log doesn't show a 'details' or 'message' field for failures, but keep it for robustness
        const entryDetails = log_entry.details || log_entry.message || log_entry.error_message || '';

        return {
          id: String(log_entry.id || defaultId), // Use log_entry.id as per your JSON
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

  // Fetch activities when component mounts
  useEffect(() => {
    fetchRecentActivities();
    // Add a polling mechanism if activities should update frequently
    // For a manual page, this might not be strictly necessary, but good for active logs
    const intervalId = setInterval(fetchRecentActivities, 3000); // Poll every 3 seconds
    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [fetchRecentActivities]);

  // --- handleProcess Function - Adjusting for the new ManualEntry structure ---
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
      formData.append("file", selectedPhoto[0]);

      // Use the new proxied URLs
      const endpoint = selectedAction === "entry"
        ? RECOGNITION_API_ENTER_URL
        : RECOGNITION_API_EXIT_URL;

      console.log(`Sending recognition request to: ${endpoint}`);

      const response = await fetch(endpoint, {
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
        status: apiResult.success ? "success" : "failed",
        confidence: apiResult.confidence, // Use raw confidence from FastAPI
        userId: apiResult.user_id || 'unknown_id',
        userName: apiResult.user_name || 'Unknown User',
        method: 'kiosk_capture', // Manual entry is always 'kiosk_capture'
        details: apiResult.success ? undefined : (apiResult.message || 'Recognition failed.')
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

  // --- setErrorStateDuringProcessing - Also updated for new ManualEntry structure ---
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
        return <ArrowRight className="h-4 w-4 text-primary" />;
      case "exit":
        return <ArrowLeft className="h-4 w-4 text-orange-500" />;
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
        return "text-primary border-primary/30 bg-primary/10";
      case "exit":
        return "text-orange-500 border-orange-500/30 bg-orange-500/10";
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
        return <CheckCircle className="h-4 w-4 text-primary" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
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
    <div className="min-h-full bg-gradient-primary">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        {/* Header - This existing header content will now appear below TopButtons */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-gradient-neon bg-clip-text text-transparent mb-2">
            Manual Recognition
          </h1>
          <p className="text-muted-foreground">
            Process entry/exit manually using photo upload
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Processing Panel */}
          <div className="space-y-6">
            {/* Upload Section */}
            <Card className="bg-gradient-card border-primary/30 p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <Upload className="h-5 w-5 text-primary" />
                <span>Photo Upload</span>
              </h2>

              <FileUpload
                accept="image/*"
                label="Select Face Photo"
                onFileSelect={handlePhotoSelect}
                maxSize={10}
              />
            </Card>

            {/* Action Selection */}
            <Card className="bg-gradient-card border-primary/30 p-6">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <User className="h-5 w-5 text-primary" />
                <span>Action Selection</span>
              </h2>

              <div className="space-y-4">
                <Select
                  value={selectedAction}
                  onValueChange={(value) => setSelectedAction(value as "entry" | "exit" | "")}
                >
                  <SelectTrigger className="bg-input border-border">
                    <SelectValue placeholder="Select action" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="entry">
                      <div className="flex items-center space-x-2">
                        <LogIn className="h-4 w-4 text-primary" />
                        <span>Entry</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="exit">
                      <div className="flex items-center space-x-2">
                        <LogOut className="h-4 w-4 text-orange-500" />
                        <span>Exit</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>

                <div className="flex space-x-4">
                  <Button
                    variant="cyber"
                    onClick={handleProcess}
                    disabled={!selectedPhoto.length || !selectedAction || isProcessing}
                    className="flex-1"
                  >
                    {isProcessing ? "Processing..." : "Process Recognition"}
                  </Button>

                  <Button
                    variant="outline"
                    onClick={handleReset}
                    disabled={isProcessing}
                  >
                    Reset
                  </Button>
                </div>
              </div>
            </Card>

            {/* Results Section */}
            {(result || isProcessing) && (
              <Card className="bg-gradient-card border-primary/30 p-6">
                <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-primary" />
                  <span>Results</span>
                </h2>

                {isProcessing ? (
                  <div className="text-center py-8">
                    <div className="h-12 w-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-primary font-medium">Processing recognition...</p>
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
              </Card>
            )}
          </div>

          {/* Recent Entries */}
          <Card className="bg-gradient-card border-primary/30">
            <div className="p-6 border-b border-primary/30">
              <h2 className="text-xl font-semibold flex items-center space-x-2">
                <Clock className="h-5 w-5 text-primary" />
                <span>Recent Manual Entries</span>
                {isActivitiesLoading && <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin ml-2" />}
              </h2>
            </div>

            <div className="max-h-96 overflow-y-auto">
              {isActivitiesLoading ? (
                <div className="p-8 text-center">
                  <div className="h-12 w-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
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
                    // Confidence display correction for recent entries list as well
                    const confidenceDisplay = entry.confidence !== undefined && entry.confidence !== null
                        ? `${entry.confidence.toFixed(2)}% confidence`
                        : 'N/A';

                    return (
                      <div
                        key={entry.id}
                        className={cn(
                          "group relative p-3 rounded-lg transition-all duration-200 hover:bg-primary/5",
                          index === 0 && "bg-primary/10 border border-primary/20"
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
                                    Kiosk Capture
                                  </Badge>
                                )}
                                {entry.confidence !== undefined && entry.confidence !== null && (
                                  <span className="text-primary">
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
                          <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-6 bg-primary rounded-r" />
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