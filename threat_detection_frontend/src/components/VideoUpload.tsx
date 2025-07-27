import React, { useState } from 'react';
import { CloudArrowUpIcon } from '@heroicons/react/24/outline';
import { api } from '../api/client';
import toast from 'react-hot-toast';
import { VideoProcessResponse } from '@/types/detection'; // Import the response type

interface VideoUploadProps {
  onUploadSuccess: (videoId: string) => void; // Now passes videoId
  isBeta?: boolean;
  // uploadEndpoint?: string; // This prop is no longer needed as we're hardcoding /process_video
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onUploadSuccess,
  isBeta = false,
  // uploadEndpoint = '/library/upload' // Default is now hardcoded to /process_video
}) => {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0); // 0-100, for visual progress
  const uploadEndpoint = '/process_video'; // Hardcode the endpoint

  const handleFileSelect = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      toast.error('Please select a video file');
      return;
    }

    if (file.size > 100 * 1024 * 1024) { // 100MB limit
      toast.error('File size must be less than 100MB');
      return;
    }

    setUploading(true);
    setProcessingProgress(0); // Reset progress
    const formData = new FormData();
    formData.append('video_file', file); // Backend expects 'video_file'
    formData.append('camera_id', `uploaded_video_${Date.now()}`); // Unique ID for this upload
    formData.append('camera_name', file.name); // Use filename as camera name

    try {
      // Simulate upload progress (optional, as backend processes after full upload)
      // For a true progress bar, you'd need a backend that streams upload progress.
      // Here, we'll just show an indeterminate processing state after upload.
      
      const response = await api.post<VideoProcessResponse>(uploadEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        // You would add onUploadProgress here if your axios instance was configured for it
        // onUploadProgress: (progressEvent) => {
        //   const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        //   setProcessingProgress(percentCompleted);
        // },
      });

      // Once the response is received, processing is complete.
      setProcessingProgress(100); 
      onUploadSuccess(response.data.video_id); // Pass the video_id from the response
      toast.success(`Video uploaded and processing started successfully!`);
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload and process video');
    } finally {
      setUploading(false);
      setProcessingProgress(0); // Reset for next upload
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center text-white transition-colors ${
          dragOver
            ? isBeta
              ? 'border-purple-400 bg-purple-50'
              : 'border-primary-light bg-primary-lighter'
            : 'border-border hover:border-foreground'
        }`}
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
      >
        <CloudArrowUpIcon className={`mx-auto h-12 w-12 ${isBeta ? 'text-purple-400' : 'text-white'}`} />

        <div className="mt-4">
          <h3 className="text-lg font-medium text-foreground">
            Upload {isBeta ? 'Beta ' : ''}Video
          </h3>
          <p className="text-muted-foreground mt-1">
            Drag and drop your soccer video here, or click to browse
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Supports MP4 files up to 100MB
          </p>
        </div>

        <div className="mt-6">
          <label className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white ${
            isBeta
              ? 'bg-purple-600 hover:bg-purple-700'
              : 'bg-[#36D399]'
          } focus:outline-none focus:ring-2 focus:ring-offset-2 ${
            isBeta ? 'focus:ring-purple-500' : 'focus:ring-primary'
          } cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed`}>
            {uploading ? 'Uploading...' : 'Choose File'}
            <input
              type="file"
              className="sr-only"
              accept="video/*"
              onChange={handleFileInput}
              disabled={uploading}
            />
          </label>
        </div>

        {uploading && (
          <div className="mt-4">
            <div className="w-full bg-muted rounded-full h-2">
              {/* Indeterminate progress bar for processing */}
              <div className={`h-2 rounded-full ${
                isBeta ? 'bg-purple-600' : 'bg-[#36D399]'
              } animate-pulse`} style={{ width: '100%' }}></div>
            </div>
            <p className="text-sm text-muted-foreground mt-2">Processing video...</p>
          </div>
        )}
      </div>
    </div>
  );
};
