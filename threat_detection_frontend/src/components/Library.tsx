import React, { useState, useEffect } from 'react';
import { VideoCard } from './VideoCard';
import { VideoUpload } from './VideoUpload';
import { VideoModal } from './VideoModal';
import { api } from '../api/client';
import toast from 'react-hot-toast';
import { Video } from '@/types/video';


const Library: React.FC = () => {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const fetchVideos = async () => {
    try {
      // Assuming /library endpoint will eventually return video metadata
      // including whether it has processed output (has_output).
      // For now, it's a placeholder.
      // The backend's /process_video doesn't add to a /library list directly.
      // This /library endpoint would need to be implemented on the backend.
      // For this demo, we'll simulate a fetch.
      // A more robust solution would involve a backend endpoint that lists all processed videos.
      // For now, this will fetch a dummy list or rely on the backend's /library if it exists.
      const response = await api.get('/library'); // This endpoint needs to be implemented on backend
      setVideos(response.data);
    } catch (error) {
      toast.error('Failed to load videos from library');
      console.error('Error fetching videos:', error);
      setVideos([]); // Ensure videos array is empty on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial fetch of videos when component mounts
    fetchVideos();
  }, []);

  const handleVideoClick = (video: Video) => {
    setSelectedVideo(video);
    setModalOpen(true);
  };

  // The onUploadSuccess now receives the video_id from the backend
  const handleUploadSuccess = (videoId: string) => {
    // Instead of directly adding, we'll refetch to get the full video object
    // with its updated status (e.g., 'processing' or 'completed' and has_output)
    fetchVideos();
    toast.success(`Video uploaded and processing initiated for ID: ${videoId}!`);
  };

  const handleVideoUpdate = (updatedVideo: Video) => {
    setVideos(videos.map(v => v.id === updatedVideo.id ? updatedVideo : v));
    if (selectedVideo && selectedVideo.id === updatedVideo.id) {
      setSelectedVideo(updatedVideo);
    }
  };

  const handleVideoDelete = (videoId: string) => {
    setVideos(videos.filter(v => v.id !== videoId));
    if (selectedVideo && selectedVideo.id === videoId) {
      setModalOpen(false);
      setSelectedVideo(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-64 text-foreground">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg shadow-surface">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Video Library</h2>
          <p className="text-muted-foreground mt-1">
            Manage your uploaded videos and review analysis results.
          </p>
        </div>
      </div>

      {/* Pass the updated handleUploadSuccess */}
      <VideoUpload onUploadSuccess={handleUploadSuccess} />

      {videos.length === 0 ? (
        <div className="text-center py-12">
          <div className="mx-auto h-24 w-24 text-muted-foreground mb-4 opacity-50">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">No videos yet</h3>
          <p className="text-muted-foreground">Upload your first video to get started.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {videos.map((video) => (
            <VideoCard
              key={video.id}
              video={video}
              onClick={() => handleVideoClick(video)}
            />
          ))}
        </div>
      )}

      {selectedVideo && (
        <VideoModal
          video={selectedVideo}
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          onVideoUpdate={handleVideoUpdate}
          onVideoDelete={handleVideoDelete}
        />
      )}
    </div>
  );
};

export default Library;
