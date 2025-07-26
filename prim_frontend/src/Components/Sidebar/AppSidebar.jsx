import { useState } from "react";
import { Monitor, User, Shield, Users, MapPin, ChevronLeft, ChevronRight } from "lucide-react";

const navigationItems = [
  { title: "Dashboard", url: "/dashboard", icon: Monitor, description: "Main control center", isActive: true },
  { title: "Face ID", url: "http://localhost:8080/monitor", icon: User, description: "Facial recognition system" },
  { title: "Weapon Detect", url: "/http://localhost:8081/", icon: Shield, description: "Weapon identification" },
  { title: "Riot Monitor", url: "/riot-monitor", icon: Users, description: "Riot detection system" },
  { title: "Area Guard", url: "/area-guard", icon: MapPin, description: "Restricted area monitoring" },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  const handleNavigation = (url) => {
    window.location.href = url;
  };

  return (
    <div className={`h-screen transition-all duration-300 ${collapsed ? "w-16" : "w-64"} bg-[#101921] border-r border-zinc-700 p-4 text-white relative`}>
      {/* Collapse Toggle Button */}
      <button
        className="absolute -right-3 top-5 bg-[#101921] border border-[#424953] rounded-full p-1 z-50"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      {/* Sidebar Content */}
      {!collapsed && (
        <>
          <div className="text-xl font-semibold mb-6 px-2">Detection Systems</div>
          <div className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.title}
                  onClick={() => handleNavigation(item.url)}
                  className={`
                    flex items-center space-x-4 p-3 rounded-lg cursor-pointer transition-all
                    ${item.isActive
                      ? "bg-[zinc-700] text-white shadow-lg border border-zinc-600"
                      : "hover:bg-zinc-700 hover:text-white text-zinc-400"}
                  `}
                >
                  <div className={`
                    p-2 rounded-md
                    ${item.isActive ? "bg-zinc-600" : "bg-[#1F2733] group-hover:bg-zinc-700"}
                  `}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <div className="font-medium">{item.title}</div>
                    <div className="text-sm text-zinc-400">{item.description}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
