import { Shield, Eye, Target, Users, MapPin } from "lucide-react";
import { useState } from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";

const detectionSystems = [
  {
    title: "Dashboard",
    description: "Main control center",
    url: import.meta.env.VITE_APP_DASHBOARD_URL || "http://localhost:3000",
    icon: Shield,
    isActive: false, // Current app
  },
  {
    title: "Face ID",
    description: "Facial recognition system",
    url: import.meta.env.VITE_APP_FACE_ID_URL || "http://localhost:8081",
    icon: Eye,
    isActive: true,
  },
  {
    title: "Weapon Detect",
    description: "Weapon identification",
    url: import.meta.env.VITE_APP_WEAPON_DETECT_URL || "http://localhost:8080",
    icon: Target,
    isActive: false,
  },
  {
    title: "Riot Monitor",
    description: "Riot detection system",
    url: import.meta.env.VITE_APP_RIOT_MONITOR_URL || "http://localhost:3003",
    icon: Users,
    isActive: false,
  },
  {
    title: "Area Guard",
    description: "Restricted area monitoring",
    url: import.meta.env.VITE_APP_AREA_GUARD_URL || "http://localhost:3004",
    icon: MapPin,
    isActive: false,
  },
];

export function AppSidebar() {
  const { state } = useSidebar();

  const handleNavigation = (url: string, isActive: boolean) => {
    if (!isActive) {
      window.location.href = url;
    }
  };

  return (
    <Sidebar className="border-r border-primary/20 bg-gradient-to-b from-card/50 to-card/30">
      <SidebarContent className="py-6">
        <SidebarGroup>
          <SidebarGroupLabel className="text-primary text-lg font-semibold mb-4 px-4">
            Detection Systems
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-2">
              {detectionSystems.map((system) => {
                const Icon = system.icon;
                return (
                  <SidebarMenuItem key={system.title}>
                    <SidebarMenuButton
                      onClick={() => handleNavigation(system.url, system.isActive)}
                      className={`
                        h-16 px-4 rounded-lg transition-all duration-300 cursor-pointer group
                        ${system.isActive 
                          ? 'bg-primary/20 border border-primary/50 shadow-glow-subtle text-primary' 
                          : 'hover:bg-primary/10 hover:border-primary/30 text-muted-foreground hover:text-primary'
                        }
                      `}
                      asChild
                    >
                      <div className="flex items-center space-x-4 w-full">
                        <div className={`
                          p-2 rounded-md transition-colors
                          ${system.isActive 
                            ? 'bg-primary/30 text-primary' 
                            : 'bg-muted/20 text-muted-foreground group-hover:bg-primary/20 group-hover:text-primary'
                          }
                        `}>
                          <Icon className="h-5 w-5" />
                        </div>
                        {state !== "collapsed" && (
                          <div className="flex-1 text-left">
                            <div className="font-medium text-base">
                              {system.title}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {system.description}
                            </div>
                          </div>
                        )}
                      </div>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}