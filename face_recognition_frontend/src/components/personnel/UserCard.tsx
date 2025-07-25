import { User, Clock, MapPin } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface UserCardProps {
  user: {
    id: string;
    name: string;
    role: "student" | "professor" | "guard" | "maintenance";
    status: "on-site" | "off-site";
    photo?: string;
    lastSeen?: string;
    department?: string;
  };
  className?: string;
}

const roleColors = {
  student: "border-blue-500/50 bg-blue-500/10",
  professor: "border-purple-500/50 bg-purple-500/10", 
  guard: "border-orange-500/50 bg-orange-500/10",
  maintenance: "border-yellow-500/50 bg-yellow-500/10",
};

const roleLabels = {
  student: "Student",
  professor: "Professor",
  guard: "Security Guard",
  maintenance: "Maintenance",
};

export const UserCard = ({ user, className }: UserCardProps) => {
  const isOnSite = user.status === "on-site";

  return (
    <Card className={cn(
      "group relative overflow-hidden bg-gradient-card border transition-all duration-300 hover:shadow-glow-subtle",
      roleColors[user.role],
      className
    )}>
      {/* Status Indicator */}
      <div className="absolute top-3 right-3 z-10">
        <div className={cn(
          "h-3 w-3 rounded-full",
          isOnSite 
            ? "bg-primary shadow-glow-subtle animate-pulse" 
            : "bg-muted-foreground"
        )} />
      </div>

      <div className="p-6">
        {/* Avatar */}
        <div className="relative mb-4 mx-auto w-20 h-20">
          {user.photo ? (
            <img
              src={user.photo}
              alt={user.name}
              className="w-full h-full rounded-full object-cover border-2 border-primary/30"
            />
          ) : (
            <div className="w-full h-full rounded-full bg-muted flex items-center justify-center border-2 border-primary/30">
              <User className="h-8 w-8 text-muted-foreground" />
            </div>
          )}
          
          {/* Online indicator overlay */}
          {isOnSite && (
            <div className="absolute -bottom-1 -right-1">
              <div className="h-6 w-6 bg-primary rounded-full border-2 border-card flex items-center justify-center">
                <div className="h-2 w-2 bg-primary-foreground rounded-full" />
              </div>
            </div>
          )}
        </div>

        {/* User Info */}
        <div className="text-center space-y-3">
          <div>
            <h3 className="font-semibold text-lg text-foreground group-hover:text-primary transition-colors">
              {user.name}
            </h3>
            <p className="text-sm text-muted-foreground">{user.id}</p>
          </div>

          {/* Role Badge */}
          <Badge 
            variant="secondary" 
            className={cn(
              "capitalize border",
              roleColors[user.role]
            )}
          >
            {roleLabels[user.role]}
          </Badge>

          {/* Department */}
          {user.department && (
            <div className="flex items-center justify-center space-x-1 text-sm text-muted-foreground">
              <MapPin className="h-3 w-3" />
              <span>{user.department}</span>
            </div>
          )}

          {/* Status */}
          <div className="flex items-center justify-center space-x-2">
            <div className={cn(
              "h-2 w-2 rounded-full",
              isOnSite ? "bg-primary" : "bg-muted-foreground"
            )} />
            <span className={cn(
              "text-sm font-medium",
              isOnSite ? "text-primary" : "text-muted-foreground"
            )}>
              {isOnSite ? "On Site" : "Off Site"}
            </span>
          </div>

          {/* Last Seen */}
          {user.lastSeen && (
            <div className="flex items-center justify-center space-x-1 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              <span>Last seen: {user.lastSeen}</span>
            </div>
          )}
        </div>
      </div>

      {/* Hover Effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
    </Card>
  );
};