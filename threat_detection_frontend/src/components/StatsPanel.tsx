import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity, Camera, Shield, Clock } from 'lucide-react';
import { AppStats } from '@/types/detection'; // Assuming AppStats type matches the backend response structure
import { config } from '@/config/env';
import { cn } from '@/lib/utils';

interface StatsPanelProps {
  activeCameraCount: number;
  totalThreats: number;
}

export function StatsPanel({ activeCameraCount, totalThreats }: StatsPanelProps) {
  const [stats, setStats] = useState<AppStats | null>(null);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${config.BACKEND_HTTP_URL}/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    // Ensure leading zeros for minutes if less than 10
    return `${hours}h ${mins.toString().padStart(2, '0')}m`;
  };

  const statCards = [
    {
      label: 'Active Cameras',
      value: activeCameraCount,
      icon: Camera,
      color: 'primary'
    },
    {
      label: 'Total Threats',
      value: totalThreats,
      icon: Shield,
      color: totalThreats > 0 ? 'threat' : 'safe'
    },
    {
      label: 'Total Detections',
      // CORRECTED: Access stats.performance.total_detections
      value: stats?.performance?.total_detections || 0,
      icon: Activity,
      color: 'primary'
    },
    {
      label: 'Uptime',
      // CORRECTED: Access stats.performance.uptime_seconds
      value: stats?.performance?.uptime_seconds ? formatUptime(stats.performance.uptime_seconds) : '0h 0m',
      icon: Clock,
      color: 'primary'
    }
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {statCards.map((stat) => {
        const IconComponent = stat.icon;
        return (
          <Card key={stat.label} className="bg-gradient-card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-white">{stat.label}</p>
                <p className={cn(
                  "text-2xl font-bold",
                  `text-[#A1A4A6]` // Apply color from stat.color
                )}>
                  {stat.value}
                </p>
              </div>
              <div className={cn(
                "p-2 rounded-lg",
                `bg-[#101921]` // Apply background color from stat.color
              )}>
                <IconComponent className={cn(
                  "h-5 w-5",
                  `text-[white]` // Apply text color from stat.color
                )} />
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
}
