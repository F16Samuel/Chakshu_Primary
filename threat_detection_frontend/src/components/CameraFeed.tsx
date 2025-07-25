import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Camera, CameraOff, AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useCamera } from '@/hooks/useCamera';
import { DetectionResult, CameraStatus } from '@/types/detection';
import { config } from '@/config/env';
import { cn } from '@/lib/utils';

interface CameraFeedProps {
  camera: CameraStatus;
  onDetection: (result: DetectionResult) => void;
  onRemove: () => void;
  onStatusChange: (status: Partial<CameraStatus>) => void;
}

export function CameraFeed({ camera, onDetection, onRemove, onStatusChange }: CameraFeedProps) {
  const { videoRef, startCamera, stopCamera, captureFrame } = useCamera();
  const [frameInterval, setFrameInterval] = useState<NodeJS.Timeout>();
  const [isActive, setIsActive] = useState(false);
  const [displayedThreatCount, setDisplayedThreatCount] = useState(0); 
  const displayedThreatTimeoutRef = useRef<NodeJS.Timeout>(); 

  console.log(`[${camera.name}] Render - threatActive: ${camera.threatActive}, displayedThreatCount: ${displayedThreatCount}, lastDetection:`, camera.lastDetection); // DEBUG

  const { status: wsStatus, sendFrame } = useWebSocket({
    cameraId: camera.id,
    onDetection: (result) => {
      console.log(`[${camera.name}] WebSocket onDetection received:`, result); // DEBUG
      // If the backend sends an error, log it but don't try to process as detection result
      if (result && 'error' in result) {
        console.error(`[${camera.name}] Backend Error: ${result.error}`);
        // Optionally, show a popup or update camera status to error
        onStatusChange({ status: 'error' });
        return; 
      }

      onDetection(result); 
      onStatusChange({ lastDetection: { ...result } }); 

      if (result.threat_detected) {
        console.log(`[${camera.name}] Threat detected in frame. Activating highlight.`); 
        onStatusChange({ threatActive: true });
        if (camera.threatTimeout) {
          clearTimeout(camera.threatTimeout);
          console.log(`[${camera.name}] Cleared existing threat highlight timeout.`); 
        }
        const timeout = setTimeout(() => {
          console.log(`[${camera.name}] Threat highlight timeout expired. Deactivating highlight.`); 
          onStatusChange({ threatActive: false, threatTimeout: undefined });
        }, config.THREAT_HIGHLIGHT_DURATION);
        onStatusChange({ threatTimeout: timeout });

        if (result.detections.length > 0) {
          console.log(`[${camera.name}] Setting displayedThreatCount to: ${result.detections.length}`); 
          setDisplayedThreatCount(result.detections.length); 

          if (displayedThreatTimeoutRef.current) {
            clearTimeout(displayedThreatTimeoutRef.current);
            console.log(`[${camera.name}] Cleared existing displayed threat count timeout.`); 
          }
          displayedThreatTimeoutRef.current = setTimeout(() => {
            console.log(`[${camera.name}] Displayed threat count timeout expired. Resetting to 0.`); 
            setDisplayedThreatCount(0); 
          }, config.THREAT_HIGHLIGHT_DURATION);
        }
      } else {
        console.log(`[${camera.name}] No threat detected in frame.`); 
        if (!displayedThreatTimeoutRef.current) {
          console.log(`[${camera.name}] No active displayed threat timeout, resetting displayedThreatCount to 0.`); 
          setDisplayedThreatCount(0);
        }
      }
    },
    enabled: isActive
  });

  const startFeed = async () => {
    try {
      onStatusChange({ status: 'initializing' });
      await startCamera(camera.deviceId); 
      setIsActive(true);
      onStatusChange({ status: 'connected' });
    } catch (error) {
      onStatusChange({ status: 'error' });
      console.error('Failed to start camera feed:', error);
    }
  };

  const stopFeed = () => {
    setIsActive(false);
    stopCamera();
    
    if (frameInterval) {
      clearInterval(frameInterval);
    }
    
    if (camera.threatTimeout) {
      clearTimeout(camera.threatTimeout);
    }
    if (displayedThreatTimeoutRef.current) {
      clearTimeout(displayedThreatTimeoutRef.current);
    }
    
    onStatusChange({ 
      status: 'disconnected',
      threatActive: false,
      threatTimeout: undefined 
    });
    setDisplayedThreatCount(0); 
    console.log(`[${camera.name}] Camera stopped. Resetting states.`); 
  };

  // Frame capture and sending loop
  useEffect(() => {
    console.log(`[${camera.name}] Frame sending effect. isActive: ${isActive}, wsStatus: ${wsStatus}`); 
    if (isActive && wsStatus === 'connected') {
      const interval = setInterval(() => {
        const frameData = captureFrame();
        if (frameData) {
          console.log(`[${camera.name}] Captured frameData length: ${frameData.length}`); 
          const base64 = frameData.split(',')[1];
          
          // FIX: Send data as a JSON object as expected by the backend
          sendFrame(JSON.stringify({
            type: 'frame',
            data: base64,
            frame_id: Date.now(), // Use Date.now() for unique frame_id
            camera_id: camera.id,
            camera_name: camera.name // Send camera name for backend logging
          }));
        } else {
          console.warn(`[${camera.name}] captureFrame returned null/empty data.`); 
        }
      }, config.FRAME_INTERVAL);

      setFrameInterval(interval);

      return () => {
        console.log(`[${camera.name}] Clearing frame sending interval.`); 
        if (interval) clearInterval(interval);
      };
    }
  }, [isActive, wsStatus, captureFrame, sendFrame, camera.id, camera.name]); // Added camera.id, camera.name to dependencies

  const getStatusColor = () => {
    switch (camera.status) {
      case 'connected': return 'safe';
      case 'initializing': return 'warning';
      case 'error': return 'threat';
      default: return 'muted';
    }
  };

  const getStatusIcon = () => {
    switch (wsStatus) {
      case 'connected': return <Wifi className="h-4 w-4" />;
      case 'connecting': return <WifiOff className="h-4 w-4 animate-pulse" />;
      default: return <WifiOff className="h-4 w-4" />;
    }
  };

  return (
    <Card className={cn(
      "relative overflow-hidden transition-all duration-300",
      "bg-gradient-card border-border/50 shadow-elevated",
      camera.threatActive && "animate-pulse-threat border-threat"
    )}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 pb-2">
        <div className="flex items-center gap-2">
          <Camera className="h-5 w-5 text-primary" />
          <h3 className="font-semibold text-foreground">{camera.name}</h3>
        </div>
        
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={cn(
            "gap-1 border-0",
            `bg-${getStatusColor()}-bg text-${getStatusColor()}`
          )}>
            {getStatusIcon()}
            {camera.status}
          </Badge>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onRemove}
            className="text-muted-foreground hover:text-destructive"
          >
            <CameraOff className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Video Feed */}
      <div className="relative aspect-video bg-secondary/20 mx-4 rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
        />
        
        {camera.status === 'disconnected' && (
          <div className="absolute inset-0 flex items-center justify-center bg-secondary/50">
            <Button onClick={startFeed} variant="outline" className="gap-2">
              <Camera className="h-4 w-4" />
              Start Camera
            </Button>
          </div>
        )}

        {camera.threatActive && (
          <div className="absolute top-2 right-2">
            <Badge className="bg-gradient-threat border-0 gap-1 animate-pulse">
              <AlertTriangle className="h-3 w-3" />
              THREAT DETECTED
            </Badge>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="p-4 pt-3">
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div className="text-center">
            <div className="text-muted-foreground">Threats</div>
            <div className={cn(
              "font-bold text-lg",
              displayedThreatCount > 0 ? "text-threat" : "text-safe"
            )}>
              {displayedThreatCount} 
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-muted-foreground">FPS</div>
            <div className="font-bold text-lg text-foreground">
              {camera.lastDetection?.fps?.toFixed(1) || '0.0'}
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-muted-foreground">Proc. Time</div>
            <div className="font-bold text-lg text-foreground">
              {camera.lastDetection?.processing_time?.toFixed(0) || '0'}ms
            </div>
          </div>
        </div>

        {camera.status === 'connected' && (
          <Button 
            onClick={stopFeed} 
            variant="outline" 
            size="sm" 
            className="w-full mt-3"
          >
            Stop Feed
          </Button>
        )}
      </div>
    </Card>
  );
}
