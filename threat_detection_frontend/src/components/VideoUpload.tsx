import React, { useState } from 'react';
import { CloudArrowUpIcon, LinkIcon, ArrowPathIcon } from '@heroicons/react/24/outline'; // Added LinkIcon
import { api } from '../api/client';
import toast from 'react-hot-toast';
import { VideoProcessResponse } from '@/types/detection'; // Import the response type

interface VideoUploadProps {
  onUploadSuccess: (videoId: string) => void; // Now passes videoId
  isBeta?: boolean;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onUploadSuccess,
  isBeta = false,
}) => {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0); // 0-100, for visual progress
  const [youtubeUrl, setYoutubeUrl] = useState(''); // New state for YouTube URL
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
      const response = await api.post<VideoProcessResponse>(uploadEndpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setProcessingProgress(100); 
      onUploadSuccess(response.data.video_id); // Pass the video_id from the response
      toast.success(`Video uploaded and processing started successfully!`);
      setYoutubeUrl(''); // Clear URL input after successful file upload
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload and process video');
    } finally {
      setUploading(false);
      setProcessingProgress(0); // Reset for next upload
    }
  };

  const handleYoutubeUrlSubmit = async () => {
    if (!youtubeUrl.trim()) {
      toast.error('Please enter a YouTube URL.');
      return;
    }

    setUploading(true);
    setProcessingProgress(0);
    const formData = new FormData();
    formData.append('video_url', youtubeUrl.trim()); // Backend expects 'video_url'
    formData.append('camera_id', `youtube_video_${Date.now()}`); // Unique ID for YouTube upload
    formData.append('camera_name', `YouTube Video (${youtubeUrl.substring(0, 30)}...)`); // Short name for UI

    try {
      const response = await api.post<VideoProcessResponse>(uploadEndpoint, formData, {
        headers: {
          // No Content-Type needed for FormData when not uploading a file, Axios handles it.
          // 'Content-Type': 'multipart/form-data', // This is implicitly handled by FormData
        },
      });

      setProcessingProgress(100);
      onUploadSuccess(response.data.video_id);
      toast.success(`YouTube video processing started successfully!`);
      setYoutubeUrl(''); // Clear URL input after successful submission
    } catch (error) {
      console.error('YouTube upload error:', error);
      toast.error(`Failed to process YouTube video: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
      setProcessingProgress(0);
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
            Drag and drop your video file here, or click to browse
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

        {/* Separator */}
        <div className="my-6 flex items-center">
          <div className="flex-grow border-t border-border"></div>
          <span className="px-3 text-muted-foreground text-sm">OR</span>
          <div className="flex-grow border-t border-border"></div>
        </div>

        {/* YouTube URL Upload Section */}
        <div className="mt-4">
          <h3 className="text-lg font-medium text-foreground flex items-center justify-center">
            <LinkIcon className={`h-6 w-6 mr-2 ${isBeta ? 'text-purple-400' : 'text-primary'}`} />
            Process from YouTube URL
          </h3>
          <p className="text-muted-foreground mt-1 text-sm">
            Paste a YouTube video link to process.
          </p>
          <div className="mt-4 flex flex-col sm:flex-row gap-3">
            <input
              type="text"
              placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              className="flex-1 border border-border rounded-md px-3 py-2 text-foreground bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={uploading}
            />
            <button
              onClick={handleYoutubeUrlSubmit}
              disabled={uploading || !youtubeUrl.trim()}
              className={`px-4 py-2 rounded-md text-white font-medium ${
                isBeta
                  ? 'bg-purple-600 hover:bg-purple-700'
                  : 'bg-primary hover:bg-primary-dark'
              } focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                isBeta ? 'focus:ring-purple-500' : 'focus:ring-primary'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <ArrowPathIcon className={`h-4 w-4 mr-2 inline-block ${uploading ? 'animate-spin' : ''}`} />
              {uploading ? 'Processing URL...' : 'Process URL'}
            </button>
          </div>
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
