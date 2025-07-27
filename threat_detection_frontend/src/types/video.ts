export interface Video {
  id: string;
  filename: string;
  original_filename: string;
  upload_date: string;
  status: 'processing' | 'completed' | 'failed';
  has_output: boolean;
  file_size: number;
  duration?: number;
}
