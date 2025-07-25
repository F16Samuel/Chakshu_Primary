import { useState, useCallback } from 'react';
import { CameraFeed } from './CameraFeed';
import { ActivityLog } from './ActivityLog';
import { CameraSelector } from './CameraSelector';
import { StatsPanel } from './StatsPanel';
import { NotificationSystem } from './NotificationSystem';
import { Shield, AlertTriangle } from 'lucide-react';
import { CameraStatus, DetectionResult } from '@/types/detection';

export function Dashboard() {
  const [cameras, setCameras] = useState<CameraStatus[]>([]);
  const [threatDetections, setThreatDetections] = useState<DetectionResult[]>([]);

  const addCamera = useCallback((deviceId: string) => {
    // Check if a camera with this deviceId already exists to prevent duplicates
    if (cameras.some(cam => cam.deviceId === deviceId)) {
      console.warn(`Camera with deviceId ${deviceId} is already added.`);
      // Optionally, you could show a user-facing notification here
      return;
    }

    const cameraId = `camera-${Date.now()}`;
    const newCamera: CameraStatus = {
      id: cameraId,
      deviceId: deviceId, 
      name: `Camera ${cameras.length + 1}`,
      status: 'disconnected', 
      lastDetection: null, 
      threatActive: false
    };

    setCameras(prev => [...prev, newCamera]);
  }, [cameras]); 

  const removeCamera = useCallback((cameraId: string) => {
    setCameras(prev => prev.filter(cam => cam.id !== cameraId));
  }, []);

  const updateCameraStatus = useCallback((cameraId: string, updates: Partial<CameraStatus>) => {
    setCameras(prev => 
      prev.map(cam => {
        if (cam.id === cameraId) {
          // Create a NEW object reference for the updated camera
          return { ...cam, ...updates }; 
        }
        return cam;
      })
    );
  }, []); // Dependency array is empty, which is fine for `setCameras` functional update

  const handleDetection = useCallback((result: DetectionResult) => {
    if (result.threat_detected) {
      setThreatDetections(prev => [...prev, result]);
    }
  }, []);

  const activeCameraCount = cameras.filter(cam => cam.status === 'connected').length;
  const totalThreats = cameras.reduce((sum, cam) => 
    sum + (cam.lastDetection?.detections?.length || 0), 0
  );

  return (
    <div className="min-h-screen bg-background p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-primary rounded-lg shadow-glow">
            <Shield className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">
              Live Weapon Detection System
            </h1>
            <p className="text-muted-foreground">
              Real-time monitoring and threat detection
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${
            activeCameraCount > 0 ? 'bg-safe shadow-glow' : 'bg-muted'
          } animate-pulse`} />
          <span className="text-sm text-muted-foreground">
            {activeCameraCount > 0 ? 'Active' : 'Standby'}
          </span>
        </div>
      </div>

      {/* Stats Panel */}
      <div className="mb-8">
        <StatsPanel 
          activeCameraCount={activeCameraCount} 
          totalThreats={totalThreats} 
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Main Content */}
        <div className="xl:col-span-3 space-y-6">
          {/* Camera Selector */}
          <CameraSelector onAddCamera={addCamera} />

          {/* Camera Feeds Grid */}
          {cameras.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {cameras.map((camera) => (
                <CameraFeed
                  key={camera.id}
                  camera={camera}
                  onDetection={handleDetection} 
                  onRemove={() => removeCamera(camera.id)}
                  onStatusChange={(updates) => updateCameraStatus(camera.id, updates)}
                />
              ))}
            </div>
          )}

          {cameras.length === 0 && (
            <div className="text-center py-16">
              <AlertTriangle className="h-16 w-16 text-muted-foreground mx-auto mb-4 opacity-50" />
              <h3 className="text-xl font-semibold text-foreground mb-2">
                No Camera Feeds Active
              </h3>
              <p className="text-muted-foreground">
                Add a camera feed to start monitoring for threats
              </p>
            </div>
          )}
        </div>

        {/* Activity Log Sidebar */}
        <div className="xl:col-span-1">
          <ActivityLog />
        </div>
      </div>

      {/* Notification System */}
      <NotificationSystem threatDetections={threatDetections} />
    </div>
  );
}
