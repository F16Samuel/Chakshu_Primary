import React from 'react';
import { Video } from '@/types/video';
import { ClockIcon, PlayIcon } from '@heroicons/react/24/outline';

interface VideoCardProps {
  video: Video;
  onClick: () => void;
  isBeta?: boolean;
}

export const VideoCard: React.FC<VideoCardProps> = ({ video, onClick, isBeta = false }) => {
  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getStatusColor = () => {
    switch (video.status) {
      case 'completed':
        return video.has_output ? 'border-safe shadow-safe-light' : 'border-destructive shadow-destructive-light';
      case 'processing':
        return 'border-warning shadow-warning-light';
      case 'failed':
        return 'border-destructive shadow-destructive-light';
      default:
        return 'border-border shadow-muted';
    }
  };

  const getStatusText = () => {
    switch (video.status) {
      case 'completed':
        return video.has_output ? 'Ready ⚽' : 'No Output';
      case 'processing':
        return 'Processing...';
      case 'failed':
        return 'Failed';
      default:
        return 'Unknown';
    }
  };

  return (
    <div
      onClick={onClick}
      className={`bg-card/80 backdrop-blur-md rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer border-2 transform hover:scale-105 ${getStatusColor()} animate-slide-up`}
    >
      <div className="aspect-video bg-gradient-to-br from-primary-light to-primary-lighter rounded-t-2xl flex items-center justify-center relative overflow-hidden">
        <div className="bg-card/80 backdrop-blur-sm rounded-full p-4 shadow-lg">
          <PlayIcon className="h-8 w-8 text-primary" />
        </div>
        {isBeta && (
          <div className="absolute top-3 right-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs px-3 py-1 rounded-full font-medium shadow-lg animate-bounce-in">
            BETA ⚗️
          </div>
        )}
      </div>

      <div className="p-4">
        <h3 className="font-semibold text-foreground truncate mb-3 text-lg">
          {video.original_filename}
        </h3>

        <div className="flex items-center justify-between text-sm text-muted-foreground mb-3">
          <div className="flex items-center bg-muted rounded-lg px-2 py-1">
            <ClockIcon className="h-4 w-4 mr-1 text-primary" />
            {formatDate(video.upload_date)}
          </div>
          <span className="bg-muted rounded-lg px-2 py-1 font-medium text-foreground">{formatFileSize(video.file_size)}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className={`text-xs px-3 py-1 rounded-full font-medium ${
            video.status === 'completed' && video.has_output
              ? 'bg-safe text-safe-foreground border border-safe-dark'
              : video.status === 'processing'
              ? 'bg-warning text-warning-foreground border border-warning-dark'
              : 'bg-destructive text-destructive-foreground border border-destructive-dark'
          }`}>
            {getStatusText()}
          </span>

          {video.duration && (
            <span className="text-xs text-muted-foreground bg-muted rounded-lg px-2 py-1 font-medium">
              {Math.round(video.duration)}s
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
