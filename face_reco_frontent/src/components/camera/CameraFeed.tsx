// src/components/camera/CameraFeed.tsx

import React, { useRef, useCallback, useState } from "react";
import Webcam from "react-webcam";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, CameraIcon, AlertCircle, Video } from "lucide-react";
import { cn } from "@/lib/utils";

// Define props for the CameraFeed component for the Kiosk model
interface CameraFeedProps {
  title: string;
  onCapture: (imageData: string) => void; // Callback when an image is captured (base64 string)
  loading?: boolean; // To show loading state during capture/API call
  disabled?: boolean; // To disable buttons (e.g., if other scanner is active)
  className?: string; // Optional for external styling
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  title,
  onCapture,
  loading = false,
  disabled = false,
  className,
}) => {
  const webcamRef = useRef<Webcam>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  // Video constraints for react-webcam
  const videoConstraints = {
    width: { ideal: 1280 }, // Prefer 1280px width
    height: { ideal: 720 }, // Prefer 720px height
    facingMode: "user", // "user" for front-facing camera, "environment" for rear-facing
  };

  // Callback to capture a screenshot
  const handleCapture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        onCapture(imageSrc); // Pass the base64 image data to the parent component
      } else {
        console.error("Failed to capture screenshot: imageSrc is null");
        setWebcamError("Failed to capture image. Please try again.");
      }
    } else {
        console.error("Webcam ref is null during capture attempt.");
        setWebcamError("Webcam not ready for capture. Please wait.");
    }
  }, [webcamRef, onCapture]);

  // Callback when webcam access is successfully granted and stream starts
  const handleUserMedia = useCallback(() => {
    setIsWebcamActive(true);
    setWebcamError(null); // Clear any previous errors
    console.log(`${title} webcam access granted.`);
  }, [title]);

  // Callback when there's an error accessing the webcam
  const handleUserMediaError = useCallback((error: any) => {
    setIsWebcamActive(false);
    console.error(`Error accessing ${title} webcam:`, error);
    if (error.name === "NotAllowedError") {
      setWebcamError("Webcam access denied. Please grant permission in your browser settings.");
    } else if (error.name === "NotFoundError") {
      setWebcamError("No webcam found. Please ensure a webcam is connected and enabled.");
    } else if (error.name === "NotReadableError") {
      setWebcamError("Webcam is already in use by another application or device. Please close other apps.");
    }
    else {
      setWebcamError(`Error accessing webcam: ${error.message}`);
    }
  }, [title]);


  return (
    <Card className={cn("flex flex-col bg-[#1F2733] border-[#1F2733]/30 shadow-lg p-0", className)}>
      {/* Header */}
      <div className="relative bg-card/50 border-b border-[#1F2733]/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <CameraIcon className="h-5 w-5 text-[#36D399]" />
              {isWebcamActive && ( // Show active indicator based on local webcam status
                <div className="absolute -top-1 -right-1 h-3 w-3 bg-[#36D399] rounded-full animate-pulse" />
              )}
            </div>
            <h3 className="font-semibold text-lg text-[#36D399]-foreground">{title}</h3>
          </div>
          
          <div className={cn(
            "px-3 py-1 rounded-full text-xs font-medium border",
            isWebcamActive
              ? "bg-[#36D399]/20 text-[#36D399] border-[#1F2733]/50 shadow-glow-subtle"
              : "bg-muted text-muted-foreground border-[#1F2733]"
          )}>
            {isWebcamActive ? "ACTIVE" : "INACTIVE"}
          </div>
        </div>
      </div>

      {/* Video Feed / Webcam Display */}
      <div className="relative flex-grow aspect-video bg-black rounded-b-lg overflow-hidden flex items-center justify-center border-t-0 border border-[#1F2733]">
        {webcamError ? (
          <div className="text-destructive text-center p-4">
            <AlertCircle className="h-12 w-12 mx-auto mb-2" />
            <p className="font-semibold">Webcam Error:</p>
            <p className="text-sm mb-2">{webcamError}</p>
            <p className="text-xs text-muted-foreground">
              Please ensure your webcam is connected, not in use by other applications, and that you've granted browser permission.
            </p>
          </div>
        ) : (
          <>
            <Webcam
              audio={false} // Disable audio as it's not needed
              ref={webcamRef}
              screenshotFormat="image/jpeg" // Format for captured images
              videoConstraints={videoConstraints}
              onUserMedia={handleUserMedia} // Callback on successful webcam access
              onUserMediaError={handleUserMediaError} // Callback on webcam access error
              mirrored={true} // Common for user-facing cameras to show a mirrored view
              className="w-full h-full object-cover" // Ensure it covers the container
            />
            {/* Overlay for "Waiting for webcam access..." */}
            {!isWebcamActive && !webcamError && (
                <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center flex-col text-white">
                    <Loader2 className="h-8 w-8 animate-spin mb-2" />
                    <p>Waiting for webcam access...</p>
                    <p className="text-sm text-muted-foreground mt-1">Please grant camera permissions.</p>
                </div>
            )}

            {/* Title Overlay (positioned at the bottom now, or remove if redundant) */}
            <div className="absolute bottom-4 left-4 bg-card/90 backdrop-blur-sm border border-[#36D399]/30 rounded-md px-3 py-1">
              <span className="text-[#36D399] font-mono text-sm font-bold">{title}</span>
            </div>

            {/* Camera Source Indicator */}
            <div className="absolute top-4 right-4 bg-card/90 backdrop-blur-sm border border-[#36D399]/30 rounded-md px-2 py-1">
              <Video className="h-4 w-4 text-[#36D399]" /> {/* Always local webcam now */}
            </div>
          </>
        )}
      </div>

      {/* Controls (Capture Button) */}
      <div className="p-4 bg-card/30 flex justify-center items-center">
        <Button
          onClick={handleCapture}
          disabled={loading || disabled || !isWebcamActive} // Disable if loading, externally disabled, or webcam not active
          className="bg-[#36D399] hover:bg-[#36D399]-dark text-[#36D399]-foreground font-bold py-2 px-6 rounded-lg shadow-md transition-all duration-200"
        >
          {loading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <CameraIcon className="mr-2 h-4 w-4" />
          )}
          {loading ? "Capturing..." : "Capture for Recognition"}
        </Button>
      </div>
    </Card>
  );
};
