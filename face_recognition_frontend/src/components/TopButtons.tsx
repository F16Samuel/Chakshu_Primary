// src/components/TopButtons.tsx
import { Link, useLocation } from "react-router-dom";
import { Shield, Activity,Eye, User, Book } from "lucide-react"; // Import icons used in navigation
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils"; // Assuming you have this utility for conditional classes

// Re-use the navigation array structure from your Header component
const navigation = [
  { name: "Dashboard", href: "/", icon: Activity },
  { name: "Register", href: "/register", icon: Shield },
  { name: "Personnel", href: "/personnel", icon: User },
  { name: "Monitor", href: "/monitor", icon: Activity },
  { name: "Manual", href: "/manual", icon: Book },
];

export function TopButtons() {
  const location = useLocation();

  return (
    <div>
      <div className="relative z-10 flex justify-between items-center p-6 border-b border-[#263140] bg-[#101921] shadow-[0_2px_10px_rgba(0,0,0,0.5)]">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Eye className="h-8 w-8 text-[#36D399] animate-pulse drop-shadow-[0_0_6px_#36D399]" />
            <div className="absolute inset-0 animate-ping">
              <Eye className="h-8 w-8 text-[#36D399] opacity-50" />
            </div>
          </div>
          <Link to="http://localhost:5173/Dashboard" className="text-2xl font-bold text-[#F2F2F2] drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]">
            Chak<span className="text-[#36D399]">shu</span>
          </Link>
        </div>
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                "flex items-center space-x-2 px-3 sm:px-4 py-2 rounded-md text-sm font-medium transition-all duration-300",
                isActive
                  ? "bg-[#101921] border border-[#00AAFF] text-[#00AAFF] px-6 py-2 rounded-xl transition-all duration-300 shadow-[0_0_10px_#00AAFF]"
                  : "bg-[#101921] border border-[#36D399] text-[#36D399] px-6 py-2 rounded-xl hover:bg-[#36D399] hover:text-[#101921] transition-all duration-300 shadow-[0_0_10px_#36D39966] hover:shadow-[0_0_15px_#36D399AA]"
              )}
            >
              <item.icon className="h-4 w-4" /> {/* Render the icon */}
              <span className="hidden sm:inline">{item.name}</span> {/* Hide text on very small screens, show icon only */}
              <span className="inline sm:hidden">{item.name.charAt(0)}</span> {/* Show first letter for very small screens */}
            </Link>
          );
        })}
      </div>
    </div>
  );
}