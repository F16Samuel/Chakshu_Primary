import { useState, useEffect, useRef, useCallback } from 'react';
import { config } from '@/config/env';
import { DetectionResult } from '@/types/detection';

interface UseWebSocketProps {
  cameraId: string;
  onDetection: (result: DetectionResult) => void;
  enabled: boolean;
}

export function useWebSocket({ cameraId, onDetection, enabled }: UseWebSocketProps) {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  // Memoize onDetection to prevent re-creation if it's a dependency in other effects
  const onDetectionRef = useRef(onDetection);
  useEffect(() => {
    onDetectionRef.current = onDetection;
  }, [onDetection]);

  const connect = useCallback(() => {
    if (!enabled) return;
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return; // Already connected or connecting
    }

    try {
      const wsUrl = `${config.BACKEND_URL}/ws/detect?camera_id=${cameraId}`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setStatus('connected');
        wsRef.current = ws;
        console.log(`[WS-${cameraId}] Connected.`); // DEBUG
      };

      ws.onmessage = (event) => {
        try {
          const result: DetectionResult = JSON.parse(event.data);
          onDetectionRef.current(result); // Use the ref to call the latest onDetection
        } catch (error) {
          console.error(`[WS-${cameraId}] Failed to parse detection result:`, error);
        }
      };

      ws.onclose = (event) => {
        setStatus('disconnected');
        wsRef.current = null;
        console.log(`[WS-${cameraId}] Disconnected. Code: ${event.code}, Reason: ${event.reason}`); // DEBUG
        
        // Auto-reconnect after 3 seconds if enabled
        if (enabled) {
          console.log(`[WS-${cameraId}] Attempting to reconnect in 3 seconds...`); // DEBUG
          reconnectTimeoutRef.current = setTimeout(() => {
            setStatus('connecting');
            connect(); // Call the memoized connect
          }, 3000);
        }
      };

      ws.onerror = (error) => {
        setStatus('error');
        console.error(`[WS-${cameraId}] WebSocket error:`, error); // DEBUG
      };

      setStatus('connecting');
      console.log(`[WS-${cameraId}] Attempting to connect to ${wsUrl}`); // DEBUG
    } catch (error) {
      setStatus('error');
      console.error(`[WS-${cameraId}] WebSocket connection error:`, error); // DEBUG
    }
  }, [enabled, cameraId]); // Dependencies for useCallback

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
      console.log(`[WS-${cameraId}] Cleared reconnect timeout.`); // DEBUG
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      console.log(`[WS-${cameraId}] Explicitly closed WebSocket.`); // DEBUG
    }
    
    setStatus('disconnected');
  }, [cameraId]); // Dependencies for useCallback

  const sendFrame = useCallback((frameData: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(frameData);
    } else {
      console.warn(`[WS-${cameraId}] Cannot send frame, WebSocket not open. Current state: ${wsRef.current?.readyState}`); // DEBUG
    }
  }, [cameraId]); // Dependencies for useCallback

  useEffect(() => {
    console.log(`[WS-${cameraId}] useWebSocket effect triggered. Enabled: ${enabled}`); // DEBUG
    if (enabled) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      console.log(`[WS-${cameraId}] useWebSocket cleanup.`); // DEBUG
      disconnect();
    };
  }, [enabled, cameraId, connect, disconnect]); // Dependencies for useEffect

  return { status, sendFrame, disconnect };
}
