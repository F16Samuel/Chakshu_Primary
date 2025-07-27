import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X, Shield, Activity,Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/", icon: Activity },
  { name: "Register", href: "/register", icon: Shield },
  { name: "Personnel", href: "/personnel", icon: Shield },
  { name: "Monitor", href: "/monitor", icon: Activity },
  { name: "Manual", href: "/manual", icon: Shield },
];

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  return (
    <header className="bg-gradient-primary border-b border-primary/30 shadow-neon sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Eye className="h-8 w-8 text-[#36D399] animate-pulse drop-shadow-[0_0_6px_#36D399]" />
              <div className="absolute inset-0 animate-ping">
                <Eye className="h-8 w-8 text-[#36D399] opacity-50" />
              </div>
            </div>
            <Link to="/" className="text-2xl font-bold text-[#F2F2F2] drop-shadow-[0_1px_1px_rgba(0,0,0,0.3)]">
              Chak<span className="text-[#36D399]">shu</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    "flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all duration-300",
                    isActive
                      ? "bg-primary/20 text-primary shadow-glow-subtle border border-primary/50"
                      : "text-muted-foreground hover:text-primary hover:bg-primary/10"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </Button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden border-t border-primary/30 py-4">
            <nav className="space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setIsOpen(false)}
                    className={cn(
                      "flex items-center space-x-3 px-4 py-3 rounded-md text-sm font-medium transition-all duration-300",
                      isActive
                        ? "bg-primary/20 text-primary shadow-glow-subtle border border-primary/50"
                        : "text-muted-foreground hover:text-primary hover:bg-primary/10"
                    )}
                  >
                    <item.icon className="h-5 w-5" />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};