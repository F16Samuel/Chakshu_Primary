import { useState, useEffect } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DetectionResult } from '@/types/detection';
import { config } from '@/config/env';
import { cn } from '@/lib/utils';

interface Notification {
  id: string;
  cameraId: string;
  cameraName: string;
  timestamp: number;
}

interface NotificationSystemProps {
  threatDetections: DetectionResult[];
}

export function NotificationSystem({ threatDetections }: NotificationSystemProps) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [cooldowns, setCooldowns] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (threatDetections.length === 0) return;

    const latestThreat = threatDetections[threatDetections.length - 1];
    
    if (latestThreat.threat_detected && !cooldowns.has(latestThreat.camera_id)) {
      const notification: Notification = {
        id: `${latestThreat.camera_id}-${Date.now()}`,
        cameraId: latestThreat.camera_id,
        cameraName: `Camera ${latestThreat.camera_id.split('-')[1] || latestThreat.camera_id}`,
        timestamp: Date.now()
      };

      setNotifications(prev => [...prev, notification]);

      // Add to cooldown
      setCooldowns(prev => new Set(prev).add(latestThreat.camera_id));

      // Remove from cooldown after duration
      setTimeout(() => {
        setCooldowns(prev => {
          const newCooldowns = new Set(prev);
          newCooldowns.delete(latestThreat.camera_id);
          return newCooldowns;
        });
      }, config.NOTIFICATION_COOLDOWN);

      // Auto-remove notification after 5 seconds
      setTimeout(() => {
        removeNotification(notification.id);
      }, 5000);
    }
  }, [threatDetections, cooldowns]);

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-sm">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={cn(
            "bg-gradient-threat border border-threat/20 rounded-lg p-4 shadow-threat",
            "animate-slide-up"
          )}
        >
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 w-8 h-8 bg-threat/20 rounded-full flex items-center justify-center">
              <AlertTriangle className="h-4 w-4 text-threat" />
            </div>
            
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-white">Threat Detected!</p>
              <p className="text-sm text-white/80">{notification.cameraName}</p>
              <p className="text-xs text-white/60 mt-1">
                {new Date(notification.timestamp).toLocaleTimeString()}
              </p>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={() => removeNotification(notification.id)}
              className="text-white/60 hover:text-white hover:bg-white/10 h-6 w-6 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}