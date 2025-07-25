import { useState, useEffect, useRef, useCallback } from 'react';

export function useCamera() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>('');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>('');
  const videoRef = useRef<HTMLVideoElement>(null);

  // Memoize startCamera, stopCamera, and captureFrame
  const startCamera = useCallback(async (deviceIdToUse?: string) => {
    try {
      setError('');
      
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: deviceIdToUse || selectedDevice, // Use provided deviceId or selectedDevice
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        }
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play(); // Ensure video starts playing
      }
      console.log(`[useCamera] Started camera with deviceId: ${deviceIdToUse || selectedDevice}`); // DEBUG
      return mediaStream;
    } catch (err) {
      setError('Failed to start camera');
      console.error('[useCamera] Error starting camera:', err); // DEBUG
      throw err;
    }
  }, [selectedDevice, stream]); // Dependencies for useCallback

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      console.log('[useCamera] Stopped camera stream.'); // DEBUG
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [stream]); // Dependencies for useCallback

  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || videoRef.current.readyState < 2) { // readyState < 2 means not enough data for current frame
      console.warn('[useCamera] Video not ready for frame capture.'); // DEBUG
      return null;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      console.error('[useCamera] Could not get 2D context for canvas.'); // DEBUG
      return null;
    }

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8); // Reduced quality for faster transfer
    // console.log(`[useCamera] Captured frame: ${imageData.length} bytes`); // DEBUG: Too verbose, enable if needed
    return imageData;
  }, []); // No dependencies needed for captureFrame as it only relies on videoRef.current

  useEffect(() => {
    const getDevices = async () => {
      try {
        // Request permission first to populate device labels
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        tempStream.getTracks().forEach(track => track.stop()); // Stop immediately after getting permission

        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
        setDevices(videoDevices);
        
        if (videoDevices.length > 0 && !selectedDevice) {
          setSelectedDevice(videoDevices[0].deviceId);
        }
        console.log('[useCamera] Enumerated camera devices:', videoDevices); // DEBUG
      } catch (err) {
        setError('Failed to access camera devices. Please ensure permissions are granted.');
        console.error('[useCamera] Error getting devices:', err); // DEBUG
      }
    };

    getDevices();
  }, [selectedDevice]); // Dependency for useEffect

  return {
    devices,
    selectedDevice,
    setSelectedDevice,
    stream,
    error,
    videoRef,
    startCamera,
    stopCamera,
    captureFrame
  };
}
