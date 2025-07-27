import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import toast from 'react-hot-toast';
import { VideoFrameDetection } from '@/types/detection';
import { format } from 'date-fns';
import { EyeIcon, RefreshCcw, VideoIcon } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils'; // Assuming you have a utility for class names

interface VideoActivityLogProps {
  videoId: string;
}

export const VideoActivityLog: React.FC<VideoActivityLogProps> = ({ videoId }) => {
  const [detectedFrames, setDetectedFrames] = useState<VideoFrameDetection[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  const fetchDetectedFrames = async () => {
    setLoading(true);
    try {
      const response = await api.get<VideoFrameDetection[]>(`/logs/video_detections/${videoId}`);
      setDetectedFrames(response.data);
      setLastRefresh(new Date());
      if (response.data.length === 0) {
        toast.info('No detected frames found for this video.');
      } else {
        toast.success(`Loaded ${response.data.length} detected frames.`);
      }
    } catch (error) {
      console.error('Error fetching detected video frames:', error);
      toast.error('Failed to load detected video frames.');
      setDetectedFrames([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (videoId) {
      fetchDetectedFrames();
      // Optionally, set an interval to refresh if the video is still processing
      // const interval = setInterval(fetchDetectedFrames, 10000); // Refresh every 10 seconds
      // return () => clearInterval(interval);
    }
  }, [videoId]); // Re-fetch if videoId changes

  const timeAgo = (date: Date) => {
    const seconds = Math.floor((new Date().getTime() - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <Card className="h-full bg-card rounded-lg shadow-surface p-6 flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-foreground flex items-center gap-2">
          <VideoIcon className="h-6 w-6 text-primary" />
          Processed Video Activity
        </h2>
        <button
          onClick={fetchDetectedFrames}
          className="text-muted-foreground hover:text-foreground transition-colors flex items-center text-sm"
          disabled={loading}
        >
          <RefreshCcw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
          {loading ? 'Refreshing...' : `Updated ${timeAgo(lastRefresh)}`}
        </button>
      </div>

      <ScrollArea className="flex-grow max-h-[600px] -mx-2 pr-2">
        {loading ? (
          <div className="flex items-center justify-center min-h-32 text-foreground">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <p className="ml-3 text-muted-foreground">Loading detected frames...</p>
          </div>
        ) : detectedFrames.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <EyeIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No detected frames to display for this video.</p>
            <p className="text-sm mt-1">Ensure the video was processed and threats were detected.</p>
          </div>
        ) : (
          <div className="space-y-6 px-2">
            {detectedFrames.map((frame) => (
              <div key={frame.frame_number} className="bg-background/50 border border-border/50 rounded-lg p-4 shadow-sm">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-lg font-semibold text-foreground">Frame: {frame.frame_number}</h3>
                  <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
                    {frame.detections.length} Detections
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Timestamp: {format(new Date(frame.timestamp), 'MMM dd, yyyy HH:mm:ss')}
                </p>
                {frame.frame_image_base64 && (
                  <img
                    src={`data:image/jpeg;base64,${frame.frame_image_base64}`}
                    alt={`Detected Frame ${frame.frame_number}`}
                    className="w-full h-auto rounded-md border border-gray-300 mb-3"
                  />
                )}
                <div className="space-y-1">
                  {frame.detections.map((detection, index) => (
                    <p key={index} className="text-sm text-foreground">
                      <span className="font-medium">{detection.label}</span>:{' '}
                      {(detection.confidence * 100).toFixed(1)}% confidence
                      <span className="ml-2 text-muted-foreground text-xs">
                        BBox: [{detection.bbox.map(b => Math.round(b)).join(', ')}]
                      </span>
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </Card>
  );
};
