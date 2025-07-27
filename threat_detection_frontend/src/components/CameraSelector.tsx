import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import { Camera, Plus } from 'lucide-react';

interface CameraSelectorProps {
  onAddCamera: (deviceId: string) => void;
}

export function CameraSelector({ onAddCamera }: CameraSelectorProps) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDevices();
  }, []);

  const getDevices = async () => {
    try {
      // Request permission first
      await navigator.mediaDevices.getUserMedia({ video: true });
      
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      
      if (videoDevices.length > 0) {
        setSelectedDevice(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error('Failed to get camera devices:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddCamera = () => {
    if (selectedDevice) {
      onAddCamera(selectedDevice);
    }
  };

  return (
    <Card className="bg-gradient-card p-6">
      <div className="flex items-center gap-3 mb-4">
        <Camera className="h-5 w-5 text-white" />
        <h2 className="text-lg font-semibold">Add Camera Feed</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Select Camera Device</label>
          <Select value={selectedDevice} onValueChange={setSelectedDevice} disabled={loading}>
            <SelectTrigger>
              <SelectValue placeholder={loading ? "Loading cameras..." : "Select a camera"} />
            </SelectTrigger>
            <SelectContent>
              {devices.map((device) => (
                <SelectItem key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId.slice(0, 8)}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button 
          onClick={handleAddCamera} 
          disabled={!selectedDevice || loading}
          className="w-full gap-2"
        >
          <Plus className="h-4 w-4" />
          Add Camera Feed
        </Button>

        {devices.length === 0 && !loading && (
          <p className="text-sm text-muted-foreground text-center">
            No camera devices found. Please ensure you have granted camera permissions.
          </p>
        )}
      </div>
    </Card>
  );
}