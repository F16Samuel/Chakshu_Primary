import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X, Shield, Activity } from "lucide-react";
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
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <Shield className="h-8 w-8 text-primary group-hover:text-accent transition-colors" />
              <div className="absolute inset-0 bg-primary/20 rounded-full blur-lg group-hover:bg-accent/30 transition-all" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold bg-gradient-neon bg-clip-text text-transparent">
                CyberGuard Campus
              </h1>
              <p className="text-xs text-muted-foreground">Access Control System</p>
            </div>
          </Link>

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