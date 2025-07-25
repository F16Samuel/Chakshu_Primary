export const config = {
  BACKEND_URL: import.meta.env.VITE_BACKEND_URL || 'ws://localhost:8005',
  BACKEND_HTTP_URL: import.meta.env.VITE_BACKEND_HTTP_URL || 'http://localhost:8005',
  FRAME_INTERVAL: parseInt(import.meta.env.VITE_FRAME_INTERVAL || '200'),
  THREAT_HIGHLIGHT_DURATION: parseInt(import.meta.env.VITE_THREAT_HIGHLIGHT_DURATION || '5000'),
  NOTIFICATION_COOLDOWN: parseInt(import.meta.env.VITE_NOTIFICATION_COOLDOWN || '5000'),
} as const;