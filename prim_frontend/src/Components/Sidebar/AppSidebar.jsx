import { useState } from "react";
import { Monitor, User, Shield, Users, MapPin, ChevronLeft, ChevronRight } from "lucide-react";

// Define navigation items using environment variables for URLs
const navigationItems = [
  { title: "Dashboard", url: "/dashboard", icon: Monitor, description: "Main control center", isActive: true },
  { title: "Face ID", url: process.env.REACT_APP_FACE_ID_URL, icon: User, description: "Facial recognition system" },
  { title: "Weapon Detect", url: process.env.REACT_APP_WEAPON_DETECT_URL, icon: Shield, description: "Weapon identification" },
  { title: "Riot Monitor", url: process.env.REACT_APP_RIOT_MONITOR_URL, icon: Users, description: "Riot detection system" },
  { title: "Area Guard", url: process.env.REACT_APP_AREA_GUARD_URL, icon: MapPin, description: "Restricted area monitoring" },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  // Handles navigation when a sidebar item is clicked
  const handleNavigation = (url) => {
    // Redirects the browser to the specified URL
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

      {/* Sidebar Content - visible only when not collapsed */}
      {!collapsed && (
        <>
          <div className="text-xl font-semibold mb-6 px-2">Detection Systems</div>
          <div className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon; // Get the icon component from the item
              return (
                <div
                  key={item.title} // Unique key for list rendering
                  onClick={() => handleNavigation(item.url)} // Handle navigation on click
                  className={`
                    flex items-center space-x-4 p-3 rounded-lg cursor-pointer transition-all
                    ${item.isActive
                      ? "bg-[zinc-700] text-white shadow-lg border border-zinc-600" // Active state styling
                      : "hover:bg-zinc-700 hover:text-white text-zinc-400"} // Inactive state styling
                  `}
                >
                  <div className={`
                    p-2 rounded-md
                    ${item.isActive ? "bg-zinc-600" : "bg-[#1F2733] group-hover:bg-zinc-700"} // Icon background styling
                  `}>
                    <Icon className="h-5 w-5" /> {/* Render the icon */}
                  </div>
                  <div>
                    <div className="font-medium">{item.title}</div> {/* Item title */}
                    <div className="text-sm text-zinc-400">{item.description}</div> {/* Item description */}
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
