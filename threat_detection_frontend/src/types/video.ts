export interface Video {
  id: string; // This will correspond to backend's video_id
  filename: string;
  original_filename: string;
  upload_date: string;
  status: 'processing' | 'completed' | 'failed';
  has_output: boolean; // Indicates if there are detected frames stored
  file_size: number;
  duration?: number;
}
