import { Monitor, User, Shield, Users, MapPin } from "lucide-react"
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
} from "@/components/ui/sidebar"

const navigationItems = [
  {
    title: "Dashboard",
    url: import.meta.env.VITE_APP_DASHBOARD_URL,
    icon: Monitor,
    description: "Main control center"
  },
  {
    title: "Face ID",
    url: import.meta.env.VITE_APP_FACE_ID_URL,
    icon: User,
    description: "Facial recognition system"
  },
  {
    title: "Weapon Detect",
    url: import.meta.env.VITE_APP_WEAPON_DETECT_URL,
    icon: Shield,
    description: "Weapon identification",
    isActive: true // Current server
  },
  {
    title: "Riot Monitor",
    url: import.meta.env.VITE_APP_RIOT_MONITOR_URL,
    icon: Users,
    description: "Riot detection system"
  },
  {
    title: "Area Guard",
    url: import.meta.env.VITE_APP_AREA_GUARD_URL,
    icon: MapPin,
    description: "Restricted area monitoring"
  },
]

export function AppSidebar() {
  const { state } = useSidebar()
  const collapsed = state === "collapsed"

  const handleNavigation = (url: string) => {
    // You might want to add a check here if the URL is defined,
    // although if your env variables are set up correctly, they should be.
    if (url) {
      window.location.href = url
    } else {
      console.warn("Navigation URL is undefined for item:", url);
    }
  }

  return (
    <Sidebar className="border-r border-primary/20 bg-gradient-to-b from-card/50 to-card/30">
      <SidebarContent className="py-6">
        <SidebarGroup>
          <SidebarGroupLabel className="text-primary text-lg font-semibold mb-4 px-4">
            Detection Systems
          </SidebarGroupLabel>

          <SidebarGroupContent>
            <SidebarMenu className="space-y-2">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      onClick={() => handleNavigation(item.url)}
                      className={`
                        h-16 px-4 rounded-lg transition-all duration-300 cursor-pointer group
                        ${item.isActive 
                          ? 'bg-primary/20 border border-primary/50 shadow-glow-subtle text-primary' 
                          : 'hover:bg-primary/10 hover:border-primary/30 text-muted-foreground hover:text-primary'
                        }
                      `}
                      asChild
                    >
                      <div className="flex items-center space-x-4 w-full">
                        <div className={`
                          p-2 rounded-md transition-colors
                          ${item.isActive 
                            ? 'bg-primary/30 text-primary' 
                            : 'bg-muted/20 text-muted-foreground group-hover:bg-primary/20 group-hover:text-primary'
                          }
                        `}>
                          <Icon className="h-5 w-5" />
                        </div>
                        {state !== "collapsed" && (
                          <div className="flex-1 text-left">
                            <div className="font-medium text-base">
                              {item.title}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {item.description}
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
  )
}