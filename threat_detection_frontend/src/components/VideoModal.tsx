import React, { useState } from 'react';
import { Video } from '@/types/video';
import {
  XMarkIcon,
  PlayIcon,
  TrashIcon,
  PencilIcon,
  ArrowPathIcon,
  EyeIcon // New icon for viewing detected frames
} from '@heroicons/react/24/outline';
import { api } from '../api/client';
import toast from 'react-hot-toast';
import { VideoFrameDetection } from '@/types/detection'; // Import the new type
import { VideoActivityLog } from './VideoActivityLog'; // Import the new component

interface VideoModalProps {
  video: Video;
  isOpen: boolean;
  onClose: () => void;
  onVideoUpdate: (video: Video) => void;
  onVideoDelete: (videoId: string) => void;
  isBeta?: boolean;
}

export const VideoModal: React.FC<VideoModalProps> = ({
  video,
  isOpen,
  onClose,
  onVideoUpdate,
  onVideoDelete,
  isBeta = false
}) => {
  const [isRenaming, setIsRenaming] = useState(false);
  const [newName, setNewName] = useState(video.original_filename);
  const [processing, setProcessing] = useState(false);
  const [showDetectedFramesLog, setShowDetectedFramesLog] = useState(false); // New state to toggle the log

  if (!isOpen) return null;

  const handleRename = async () => {
    try {
      const response = await api.patch(`/library/${video.id}/rename`, {
        filename: newName
      });
      onVideoUpdate(response.data);
      setIsRenaming(false);
      toast.success('Video renamed successfully');
    } catch (error) {
      toast.error('Failed to rename video');
    }
  };

  const handleDelete = async () => {
    // Replaced window.confirm with a custom modal/dialog if needed in a real app
    if (!confirm('Are you sure you want to delete this video?')) return;

    try {
      // Assuming a delete endpoint for video metadata and its detections
      // This is a placeholder, your backend needs to implement this.
      // For now, we'll assume deleting the video also clears its detections.
      await api.delete(`/library/${video.id}`); // This endpoint doesn't exist yet, but conceptually it would handle deletion
      onVideoDelete(video.id);
      toast.success('Video deleted successfully');
    } catch (error) {
      toast.error('Failed to delete video');
    }
  };

  const handleRunModel = async () => {
    setProcessing(true);
    setShowDetectedFramesLog(false); // Hide the log if re-running
    try {
      // The backend's /process_video endpoint is a POST request that takes a file or URL
      // For existing videos, you'd typically have a re-processing endpoint.
      // Since the backend's /process_video endpoint is designed for new uploads,
      // we'll simulate re-processing by calling it with the existing video ID
      // and assuming the backend can fetch the video by ID (which it currently doesn't,
      // but this is the logical next step for a full feature).
      // For this demo, we'll just indicate processing started.
      toast.error("Re-running model on existing video is not yet implemented on the backend's /process_video endpoint. Please re-upload the video.");
      // In a real scenario, you'd have an endpoint like:
      // const response = await api.post(`/process_video_by_id/${video.id}`);
      // onVideoUpdate(response.data); // Update video status to 'processing'
      // toast.success('Processing started successfully');
    } catch (error) {
      toast.error('Failed to start processing');
    } finally {
      setProcessing(false);
    }
  };

  const handleViewDetectedFrames = () => {
    setShowDetectedFramesLog(prev => !prev); // Toggle visibility
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 transition-opacity bg-background/75" onClick={onClose}></div>

        <div className="inline-block w-full max-w-4xl p-6 my-8 overflow-hidden text-left align-middle transition-all transform bg-card shadow-xl rounded-lg">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex-1">
              {isRenaming ? (
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    className="flex-1 border border-border rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <button
                    onClick={handleRename}
                    className="px-3 py-1 bg-primary text-white rounded text-sm hover:bg-primary-dark"
                  >
                    Save
                  </button>
                  <button
                    onClick={() => setIsRenaming(false)}
                    className="px-3 py-1 bg-muted text-foreground rounded text-sm hover:bg-muted-foreground"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <h2 className="text-xl font-semibold text-foreground">
                    {video.original_filename}
                  </h2>
                  <button
                    onClick={() => setIsRenaming(true)}
                    className="p-1 text-muted-foreground hover:text-foreground"
                  >
                    <PencilIcon className="h-4 w-4" />
                  </button>
                  {isBeta && (
                    <span className="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded">
                      BETA
                    </span>
                  )}
                </div>
              )}
            </div>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          {/* Video Info */}
          <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
            <div>
              <span className="text-muted-foreground">Upload Date:</span>
              <span className="ml-2 text-foreground">{new Date(video.upload_date).toLocaleString()}</span>
            </div>
            <div>
              <span className="text-muted-foreground">File Size:</span>
              <span className="ml-2 text-foreground">{(video.file_size / (1024 * 1024)).toFixed(1)} MB</span>
            </div>
            <div>
              <span className="text-muted-foreground">Status:</span>
              <span className={`ml-2 px-2 py-1 rounded text-xs ${
                video.status === 'completed' && video.has_output
                  ? 'bg-safe text-safe-foreground'
                  : video.status === 'processing'
                  ? 'bg-warning text-warning-foreground'
                  : 'bg-destructive text-destructive-foreground'
              }`}>
                {video.status === 'completed' && video.has_output
                  ? 'Ready'
                  : video.status === 'processing'
                  ? 'Processing'
                  : video.status === 'failed'
                  ? 'Failed'
                  : 'No Output'}
              </span>
            </div>
            {video.duration && (
              <div>
                <span className="text-muted-foreground">Duration:</span>
                <span className="ml-2 text-foreground">{Math.round(video.duration)}s</span>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex space-x-2">
              {/* The current backend /process_video is for new uploads, not re-processing existing IDs */}
              {/* This button is kept for conceptual future expansion or if backend changes */}
              {(video.status === 'completed' || video.status === 'failed') && (
                <button
                  onClick={handleRunModel}
                  disabled={processing}
                  className={`flex items-center px-4 py-2 rounded text-white ${
                    isBeta
                      ? 'bg-purple-600 hover:bg-purple-700'
                      : 'bg-primary hover:bg-primary-dark'
                  } disabled:opacity-50`}
                >
                  <ArrowPathIcon className={`h-4 w-4 mr-2 ${processing ? 'animate-spin' : ''}`} />
                  {processing ? 'Running...' : 'Run Model (Re-upload needed)'}
                </button>
              )}

              {/* Button to view detected frames */}
              {video.status === 'completed' && video.has_output && (
                <button
                  onClick={handleViewDetectedFrames}
                  className={`flex items-center px-4 py-2 rounded text-white ${
                    showDetectedFramesLog ? 'bg-gray-600' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <EyeIcon className={`h-4 w-4 mr-2`} />
                  {showDetectedFramesLog ? 'Hide Detected Frames' : 'View Detected Frames'}
                </button>
              )}
            </div>

            <button
              onClick={handleDelete}
              className="flex items-center px-4 py-2 bg-destructive text-white rounded hover:bg-destructive-dark"
            >
              <TrashIcon className="h-4 w-4 mr-2" />
              Delete
            </button>
          </div>

          {/* Render VideoActivityLog if toggled */}
          {showDetectedFramesLog && video.id && (
            <div className="mt-6">
              <VideoActivityLog videoId={video.id} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
