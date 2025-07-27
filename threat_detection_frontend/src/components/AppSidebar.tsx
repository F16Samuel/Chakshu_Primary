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
    isActive: false,
  },
  {
    title: "Face ID",
    description: "Facial recognition system",
    url: import.meta.env.VITE_APP_FACE_ID_URL || "http://localhost:8081",
    icon: Eye,
    isActive: false,
  },
  {
    title: "Weapon Detect",
    description: "Weapon identification",
    url: import.meta.env.VITE_APP_WEAPON_DETECT_URL || "http://localhost:8080",
    icon: Target,
    isActive: true,
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
    <Sidebar className="border-r bg-[#101921] border border-[#424953] from-card/50 to-card/30">
      <SidebarContent className="py-6 bg-[#101921]">
        <SidebarGroup>
          <SidebarGroupLabel className="text-xl font-semibold mb-6 px-2 text-white">
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
                        h-17 px-4 rounded-lg transition-all duration-300 cursor-pointer group
                        ${system.isActive 
                          ? 'bg-[#101921] border border-[#3F3F47] text-white hover:bg-[#101921] hover:text-white' 
                          : 'hover:bg-zinc-600 hover:border-[zinc-600] text-[#A6A6A6] hover:text-white'
                        }
                      `}
                      asChild
                    >
                      <div className="flex items-center space-x-4 w-full">
                        <div className={`
                          p-2 rounded-md transition-colors
                          ${system.isActive 
                            ? 'bg-[#52525C] text-white' 
                            : 'bg-muted/20 text-[#A6A6A6]'
                          }
                        `}>
                          <Icon className="h-5 w-5" />
                        </div>
                        {state !== "collapsed" && (
                          <div className="flex-1 text-left">
                            <div className="font-medium text-base">
                              {system.title}
                            </div>
                            <div className="text-sm text-[#A6A6A6]">
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