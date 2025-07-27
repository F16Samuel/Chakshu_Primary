export interface Video {
  id: string;
  filename: string; // The name of the file on disk (e.g., uuid.mp4)
  original_filename: string; // The original name provided by the user/source
  upload_date: string; // ISO string format
  status: 'processing' | 'completed' | 'failed';
  has_output: boolean; // True if detections were found and logged
  file_size: number; // Size in bytes
  duration?: number; // Duration in seconds
  stored_path?: string; // Path on the backend server (not directly used by frontend for display)
}
