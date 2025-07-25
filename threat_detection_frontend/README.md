# Live Weapon Detection Frontend

A comprehensive web-based frontend application for real-time weapon detection monitoring. This application provides an intuitive dashboard to manage multiple camera feeds, display threat alerts, and maintain detailed activity logs.

## Features

- **Real-time Camera Monitoring**: Multiple simultaneous camera feeds with live weapon detection
- **Threat Detection & Alerts**: Visual and notification-based threat alerts with confidence scoring
- **Activity Logging**: Comprehensive threat entry/exit logging with timestamps
- **Responsive Design**: Modern, professional interface optimized for security monitoring
- **WebSocket Communication**: Real-time bi-directional communication with detection backend
- **Camera Management**: Easy addition and removal of camera feeds

## Technology Stack

- **Frontend**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom design system
- **UI Components**: Shadcn/ui component library
- **State Management**: React Hooks
- **Real-time Communication**: WebSocket API
- **Build Tool**: Vite

## Prerequisites

- Node.js 18+ and npm
- Modern web browser with camera access
- Live Weapon Detection Backend Service running on `localhost:8005`

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd live-weapon-detection-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your backend configuration:
   ```env
   VITE_BACKEND_URL=ws://localhost:8005
   VITE_BACKEND_HTTP_URL=http://localhost:8005
   VITE_FRAME_INTERVAL=200
   VITE_THREAT_HIGHLIGHT_DURATION=5000
   VITE_NOTIFICATION_COOLDOWN=5000
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:8080`

## Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_BACKEND_URL` | WebSocket URL for detection backend | `ws://localhost:8005` |
| `VITE_BACKEND_HTTP_URL` | HTTP URL for REST API calls | `http://localhost:8005` |
| `VITE_FRAME_INTERVAL` | Frame sending interval in milliseconds | `200` |
| `VITE_THREAT_HIGHLIGHT_DURATION` | Threat highlighting duration in milliseconds | `5000` |
| `VITE_NOTIFICATION_COOLDOWN` | Notification cooldown period in milliseconds | `5000` |

## Usage

### Adding Camera Feeds

1. Click on the camera selector dropdown
2. Choose an available camera device
3. Click "Add Camera Feed"
4. The camera will initialize and begin streaming

### Monitoring Threats

- Camera feeds display real-time video with detection overlays
- Threat detection triggers visual highlighting and notifications
- Activity log shows detailed threat entry/exit events
- Statistics panel provides system-wide monitoring metrics

### Managing Cameras

- Use the stop button on each camera feed to disconnect
- Remove cameras using the camera-off button in the header
- Multiple cameras can be monitored simultaneously

## Backend API Integration

The frontend expects the following backend endpoints:

- `GET /health` - Health status check
- `GET /stats` - Performance statistics
- `GET /logs/threats` - Threat activity logs
- `WebSocket /ws/detect?camera_id={id}` - Real-time detection stream

## Development

### Project Structure

```
src/
├── components/          # React components
│   ├── ui/             # Shadcn UI components
│   ├── Dashboard.tsx   # Main dashboard
│   ├── CameraFeed.tsx  # Individual camera feed
│   ├── ActivityLog.tsx # Threat activity log
│   └── ...
├── hooks/              # Custom React hooks
├── types/              # TypeScript type definitions
├── config/             # Configuration files
└── lib/                # Utility functions
```

### Key Components

- **Dashboard**: Main application layout and state management
- **CameraFeed**: Individual camera stream with detection display
- **ActivityLog**: Real-time threat event logging
- **NotificationSystem**: Toast notifications for threat alerts
- **StatsPanel**: System statistics and metrics

### Custom Hooks

- **useWebSocket**: WebSocket connection management
- **useCamera**: Camera device access and frame capture

## Building for Production

```bash
npm run build
```

The built application will be available in the `dist/` directory.

## Browser Support

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

Requires camera access permissions and modern JavaScript support.

## Security Considerations

- Camera permissions are requested on first use
- WebSocket connections use secure protocols in production
- Frame data is transmitted securely to backend
- No sensitive data is stored locally

## Troubleshooting

### Camera Access Issues
- Ensure browser has camera permissions
- Check that camera is not in use by other applications
- Verify HTTPS is used in production environments

### Connection Issues
- Verify backend service is running on correct port
- Check firewall settings for WebSocket connections
- Ensure environment variables are correctly configured

### Performance Issues
- Adjust `VITE_FRAME_INTERVAL` to reduce bandwidth usage
- Monitor browser developer console for errors
- Check system resources with multiple camera feeds

## License

This project is part of the Live Weapon Detection System. Please refer to the main project license for usage terms.