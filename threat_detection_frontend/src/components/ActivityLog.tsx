import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { LogIn, LogOut, Clock, Camera, AlertTriangle } from 'lucide-react';
import { ThreatLog } from '@/types/detection';
import { config } from '@/config/env';
import { cn } from '@/lib/utils';

export function ActivityLog() {
  const [logs, setLogs] = useState<ThreatLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefreshTime, setLastRefreshTime] = useState(Date.now()); // New state for refresh time

  useEffect(() => {
    fetchLogs();
    
    // Poll for new logs every 5 seconds
    const interval = setInterval(fetchLogs, 5000);
    // Update refresh time display every 10 seconds
    const refreshTimeDisplayInterval = setInterval(updateLogRefreshTimeDisplay, 10000); 

    return () => {
      clearInterval(interval);
      clearInterval(refreshTimeDisplayInterval);
    };
  }, []);

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${config.BACKEND_HTTP_URL}/logs/threats`);
      if (response.ok) {
        const data = await response.json();
        // Ensure a new array reference is always passed to setLogs
        setLogs([...data.logs.slice(0, 50)]); // Keep only latest 50 logs, and create new array
        setLastRefreshTime(Date.now()); // Update last refresh time on successful fetch
      }
    } catch (error) {
      console.error('Failed to fetch activity logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString();
  };

  const getActionIcon = (action: string) => {
    return action === 'entry' ? LogIn : LogOut;
  };

  const getActionColor = (action: string) => {
    // These colors need to be defined in your Tailwind config or global CSS
    // For example: theme: { extend: { colors: { safe: '#48bb78', warning: '#f6ad55', threat: '#ef4444' } } }
    return action === 'entry' ? 'safe' : 'warning';
  };

  const updateLogRefreshTimeDisplay = () => {
    const refreshElement = document.getElementById('refreshLogs'); // This element is in app.py's HTML, not this React component
    if (refreshElement) {
      const now = Date.now();
      const diffMinutes = Math.floor((now - lastRefreshTime) / 60000);
      if (diffMinutes === 0) {
        refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> just now`;
      } else if (diffMinutes === 1) {
        refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> 1 min ago`;
      } else {
        refreshElement.innerHTML = `<i class="fas fa-sync-alt"></i> ${diffMinutes} mins ago`;
      }
    }
  };

  // Call updateLogRefreshTimeDisplay initially and whenever lastRefreshTime changes
  useEffect(() => {
    updateLogRefreshTimeDisplay();
  }, [lastRefreshTime]);


  if (loading) {
    return (
      <Card className="h-full bg-gradient-card">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-primary" />
            Recent Activity
          </h2>
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="h-full bg-gradient-card">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-primary" />
            Recent Activity
          </h2>
          {/* Badge for event count */}
          <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
            {logs.length} events
          </Badge>
        </div>

        <ScrollArea className="h-[500px] -mx-2">
          <div className="space-y-3 px-2">
            {logs.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No activity detected yet</p>
                <p className="text-sm mt-1">Threat events will appear here</p>
              </div>
            ) : (
              logs.map((log) => {
                const ActionIcon = getActionIcon(log.action);
                const actionColor = getActionColor(log.action);
                
                return (
                  <div
                    key={log.id}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg transition-all hover:scale-[1.02]",
                      `bg-${actionColor}-bg border border-${actionColor}/20`
                    )}
                  >
                    <div className={cn(
                      "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center",
                      `bg-${actionColor} shadow-glow`
                    )}>
                      <ActionIcon className="h-5 w-5 text-background" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <Camera className="h-4 w-4 text-muted-foreground" />
                        <span className="font-medium text-foreground">{log.camera_name}</span>
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs border-0",
                            `bg-${actionColor}/20 text-${actionColor}`
                          )}
                        >
                          {log.action}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-3 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          <span>{formatTime(log.timestamp)}</span>
                        </div>
                        
                        {log.confidence && (
                          <div className="text-xs">
                            Confidence: {(log.confidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground text-right">
                      {formatDate(log.timestamp)}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </ScrollArea>
      </div>
    </Card>
  );
}
