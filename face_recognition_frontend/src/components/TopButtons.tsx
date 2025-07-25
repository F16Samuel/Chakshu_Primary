// src/components/TopButtons.tsx
import { Link, useLocation } from "react-router-dom";
import { Shield, Activity } from "lucide-react"; // Import icons used in navigation
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils"; // Assuming you have this utility for conditional classes

// Re-use the navigation array structure from your Header component
const navigation = [
  { name: "Dashboard", href: "/", icon: Activity },
  { name: "Register", href: "/register", icon: Shield },
  { name: "Personnel", href: "/personnel", icon: Shield },
  { name: "Monitor", href: "/monitor", icon: Activity },
  { name: "Manual", href: "/manual", icon: Shield },
];

export function TopButtons() {
  const location = useLocation();

  return (
    // This div will act as your "button container" at the top of each page
    // Using a similar background and border as your header for consistency,
    // but without being a full header bar.
    <div className="flex justify-center md:justify-start space-x-2 sm:space-x-3 md:space-x-4 py-3 px-4 bg-gradient-primary border-b border-primary/30 shadow-neon-sm rounded-b-lg sticky top-0 z-40">
      {navigation.map((item) => {
        const isActive = location.pathname === item.href;
        return (
          <Link
            key={item.name}
            to={item.href}
            className={cn(
              "flex items-center space-x-2 px-3 sm:px-4 py-2 rounded-md text-sm font-medium transition-all duration-300",
              isActive
                ? "bg-primary/20 text-primary shadow-glow-subtle border border-primary/50"
                : "text-muted-foreground hover:text-primary hover:bg-primary/10"
            )}
          >
            <item.icon className="h-4 w-4" /> {/* Render the icon */}
            <span className="hidden sm:inline">{item.name}</span> {/* Hide text on very small screens, show icon only */}
            <span className="inline sm:hidden">{item.name.charAt(0)}</span> {/* Show first letter for very small screens */}
          </Link>
        );
      })}
    </div>
  );
}