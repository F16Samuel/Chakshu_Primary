// src/components/activity/ActivityFeed.tsx

import { useEffect, useRef } from "react";
import { ArrowRight, ArrowLeft, Clock, User, AlertTriangle } from "lucide-react"; // Added AlertTriangle for warnings
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Updated ActivityEntry interface to include new action types and method
interface ActivityEntry {
  id: string;
  userId: string;
  userName: string;
  action: "entry" | "exit" | "entry_attempt_already_on" | "exit_attempt_already_off"; // Added new actions
  timestamp: string;
  method: "scanner" | "kiosk_capture"; // Changed "manual" to "kiosk_capture" to match FastAPI logs
  confidence?: number;
}

interface ActivityFeedProps {
  activities: ActivityEntry[];
  className?: string;
  maxItems?: number;
}

export const ActivityFeed = ({ 
  activities, 
  className, 
  maxItems = 50 
}: ActivityFeedProps) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  console.log("DEBUG: ActivityFeed component rendering."); // Debugging: Component render

  // Auto-scroll to bottom when new activity is added
  useEffect(() => {
    if (scrollRef.current) {
      console.log("DEBUG: Auto-scrolling ActivityFeed."); // Debugging: Scroll effect
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activities]);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  };

  // Updated to handle new action types and provide appropriate icons
  const getActionIcon = (action: ActivityEntry['action']) => {
    switch (action) {
      case "entry":
        return <ArrowRight className="h-4 w-4 text-white" />;
      case "exit":
        return <ArrowLeft className="h-4 w-4 text-white" />;
      case "entry_attempt_already_on":
      case "exit_attempt_already_off":
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />; // Warning icon for attempts
      default:
        return <User className="h-4 w-4 text-muted-foreground" />; // Default icon
    }
  };

  // Updated to handle new action types and provide appropriate colors
  const getActionColor = (action: ActivityEntry['action']) => {
    switch (action) {
      case "entry":
        return "text-white border-[#1F2733] bg-[#31B184]";
      case "exit":
        return "text-white border-[#1F2733] bg-[#EB4747]";
      case "entry_attempt_already_on":
      case "exit_attempt_already_off":
        return "text-yellow-500 border-yellow-500/30 bg-yellow-500/10"; // Yellow for attempts
      default:
        return "text-muted-foreground border-muted/30 bg-muted/10"; // Default color
    }
  };

  const getActionLabel = (action: ActivityEntry['action']) => {
    switch (action) {
        case "entry": return "Entry";
        case "exit": return "Exit";
        case "entry_attempt_already_on": return "Entry Attempt (On-Site)";
        case "exit_attempt_already_off": return "Exit Attempt (Off-Site)";
        default: return "Unknown Action";
    }
  };


  const recentActivities = activities.slice(0, maxItems); // Fetch from the beginning, assuming latest are first
  console.log("DEBUG: Filtered recent activities (showing top maxItems):", recentActivities.length);

  return (
    <Card className={cn("bg-[#1F2733] border-[#1F2733]", className)}>
      <div className="p-4 border-b border-[#52525B]">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-lg flex items-center space-x-2">
            <Clock className="h-5 w-5 text-white" />
            <span>Recent Activity</span>
          </h3>
          <Badge variant="secondary" className="text-xs">
            {recentActivities.length} entries
          </Badge>
        </div>
      </div>

      <div 
        ref={scrollRef}
        className="max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-primary/20 scrollbar-track-transparent"
      >
        {recentActivities.length === 0 ? (
          <div className="p-8 text-center">
            <User className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
            <p className="text-muted-foreground">No recent activity</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {recentActivities.map((activity, index) => {
              // Debugging each activity item before rendering
              console.log(`DEBUG: Processing activity - ID: ${activity.id}, User: ${activity.userName}, Action: ${activity.action}, Confidence: ${activity.confidence}`);
              
              const isToday = new Date(activity.timestamp).toDateString() === new Date().toDateString();
              
              // Format confidence with 2 decimal places if available
              const confidenceDisplay = activity.confidence !== undefined && activity.confidence !== null
                ? `${activity.confidence.toFixed(2)}% confidence`
                : 'N/A'; // Fallback for no confidence data

              return (
                <div
                  key={activity.id}
                  className={cn(
                    "group relative p-3 rounded-lg transition-all duration-200 hover:bg-[#52525B]",
                    // Apply different background for the most recent entry
                    index === 0 && ""
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {/* Action Icon */}
                      <div className={cn(
                        "flex items-center justify-center w-8 h-8 rounded-full border",
                        getActionColor(activity.action)
                      )}>
                        {getActionIcon(activity.action)}
                      </div>

                      {/* User Info */}
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium text-sm">
                            {activity.userName || 'Unknown User'} {/* Ensure userName is displayed */}
                          </span>
                          <Badge 
                            variant="outline" 
                            className={cn(
                              "text-xs capitalize",
                              getActionColor(activity.action)
                            )}
                          >
                            {getActionLabel(activity.action)} {/* Use the new label function */}
                          </Badge>
                        </div>
                        
                        <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                          <span>ID: {activity.userId}</span>
                          {/* Changed "manual" to "kiosk_capture" */}
                          {activity.method === "kiosk_capture" && (
                            <Badge variant="secondary" className="text-xs">
                              Capture
                            </Badge>
                          )}
                          {/* Display formatted confidence */}
                          {activity.confidence !== undefined && activity.confidence !== null && (
                            <span className="text-muted-foreground">
                              {confidenceDisplay}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Timestamp */}
                    <div className="text-right text-xs text-muted-foreground">
                      <div className="font-mono">
                        {formatTime(activity.timestamp)}
                      </div>
                      {!isToday && (
                        <div className="text-xs">
                          {formatDate(activity.timestamp)}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* New entry indicator (only for the very first item, which is the most recent) */}
                  {index === 0 && (
                    <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-1 h-6" />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      
      {recentActivities.length >= maxItems && (
        <div className="p-3 border-t border-[#1F2733] text-center">
          <p className="text-xs text-muted-foreground">
            Showing last {maxItems} entries
          </p>
        </div>
      )}
    </Card>
  );
};