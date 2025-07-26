// src/pages/Personnel.tsx

import { useState, useMemo, useEffect, useCallback } from "react";
import { Search, Filter, Users, UserCheck, UserX } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { UserCard } from "@/components/personnel/UserCard";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast"; // Assuming you have a toast hook
import { TopButtons } from "@/components/TopButtons"; // Import the TopButtons component

// Use environment variable for the Node.js backend URL
const NODE_BACKEND_URL = import.meta.env.VITE_NODE_BACKEND_URL;

// Define the API URLs via the Node.js proxy
const USERS_API_URL = `${NODE_BACKEND_URL}/api/registration/users`; // Proxied through Node.js backend

// Define the structure of user data from your FastAPI backend (as received via Node.js proxy)
interface FastAPIUser {
  id_number: string;
  name: string;
  role: string;
  on_site: boolean;
  // You can add more fields if your FastAPI /users endpoint returns them
  // e.g., last_access_timestamp: string; // If you add this to your DB and expose it
}

// Define the structure for your frontend UserCard component
// It needs to match what UserCard expects (id, name, role, status, etc.)
interface FrontendUser {
  id: string;
  name: string;
  role: "student" | "professor" | "guard" | "maintenance"; // Ensure type safety for roles
  status: "on-site" | "off-site"; // Status derived from 'on_site'
  lastSeen?: string; // Optional, since we don't have it directly from DB
  department?: string; // Optional, since we don't have it directly from DB
}

const roleFilters = [
  { id: "all", label: "All Roles", icon: Users },
  { id: "student", label: "Students", icon: Users },
  { id: "professor", label: "Professors", icon: Users },
  { id: "guard", label: "Guards", icon: Users },
  { id: "maintenance", label: "Maintenance", icon: Users },
];

const statusFilters = [
  { id: "all", label: "All Status", icon: Users }, // Icon is generic Users, but UserCheck/UserX below are more specific
  { id: "on-site", label: "On Site", icon: UserCheck },
  { id: "off-site", label: "Off Site", icon: UserX },
];

export default function Personnel() {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedRole, setSelectedRole] = useState("all");
  const [selectedStatus, setSelectedStatus] = useState("all");
  const [allUsers, setAllUsers] = useState<FrontendUser[]>([]); // State to store fetched users
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Function to fetch users from the Node.js backend proxy
  const fetchUsers = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // CHANGED: Call Node.js backend proxy endpoint instead of direct FastAPI
      const response = await fetch(USERS_API_URL);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.error || "Failed to fetch users.");
      }
      const data: { users: FastAPIUser[]; total: number } = await response.json();

      // Transform FastAPIUser to FrontendUser
      const transformedUsers: FrontendUser[] = data.users.map(user => ({
        id: user.id_number,
        name: user.name,
        role: user.role as "student" | "professor" | "guard" | "maintenance", // Type assertion
        status: user.on_site ? "on-site" : "off-site",
        // 'lastSeen' and 'department' are not in your DB, so we omit them or provide placeholders
        // If you add these to your DB/API, you can map them here.
        // For 'lastSeen', you might fetch the latest access_log entry for that user.
        // For 'department', you would need to add it to your users table in SQLite.
        // For now, they will be undefined for UserCard, which should be fine if it handles optional props.
      }));

      setAllUsers(transformedUsers);
      toast({
        title: "Personnel Data Loaded",
        description: `Successfully loaded ${data.total} personnel records.`,
      });
    } catch (err) {
      console.error("Error fetching users:", err);
      setError(err instanceof Error ? err.message : "An unknown error occurred while fetching personnel.");
      toast({
        title: "Failed to Load Data",
        description: "Could not fetch personnel records. Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]); // Include toast in dependencies if it's from useToast

  // Fetch users when the component mounts
  useEffect(() => {
    fetchUsers();
  }, [fetchUsers]); // Depend on fetchUsers to avoid re-fetching unless it changes

  const filteredUsers = useMemo(() => {
    return allUsers.filter((user) => {
      const matchesSearch =
        user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        user.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (user.department && user.department.toLowerCase().includes(searchQuery.toLowerCase())); // Check department only if it exists

      const matchesRole = selectedRole === "all" || user.role === selectedRole;
      const matchesStatus = selectedStatus === "all" || user.status === selectedStatus;

      return matchesSearch && matchesRole && matchesStatus;
    });
  }, [searchQuery, selectedRole, selectedStatus, allUsers]); // Depend on allUsers

  const onSiteCount = allUsers.filter(user => user.status === "on-site").length;
  const totalCount = allUsers.length; // Use allUsers for total count

  // Adjusted counts based on current filters for display in summary
  const displayedOnSiteCount = filteredUsers.filter(user => user.status === "on-site").length;


  return (
    <div className="min-h-full bg-[#101921]">
      <TopButtons /> {/* Add the TopButtons component here */}
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-white bg-clip-text text-transparent mb-2">
            Personnel Management
          </h1>
          <p className="text-muted-foreground">
            Manage and monitor campus personnel access
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-[#1F2733] border-[#424953] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground text-sm">Total Personnel</p>
                {isLoading ? (
                    <p className="text-2xl font-bold text-foreground animate-pulse">...</p>
                ) : (
                    <p className="text-2xl font-bold text-foreground">{totalCount}</p>
                )}
              </div>
              <Users className="h-8 w-8 text-white" />
            </div>
          </Card>

          <Card className="bg-[#1F2733] border-[#424953] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground text-sm">Currently On Site</p>
                {isLoading ? (
                    <p className="text-2xl font-bold text-white animate-pulse">...</p>
                ) : (
                    <p className="text-2xl font-bold text-white">{onSiteCount}</p>
                )}
              </div>
              <UserCheck className="h-8 w-8 text-white" />
            </div>
          </Card>

          <Card className="bg-[#1F2733] border-[#424953] p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted-foreground text-sm">Off Site</p>
                {isLoading ? (
                    <p className="text-2xl font-bold text-white animate-pulse">...</p>
                ) : (
                    <p className="text-2xl font-bold text-white">{totalCount - onSiteCount}</p>
                )}
              </div>
              <UserX className="h-8 w-8 text-white" />
            </div>
          </Card>
        </div>

        {/* Search and Filters */}
        <Card className="bg-[#1F2733] border-[#424953] p-6 mb-8">
          <div className="space-y-6">
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search personnel by name or ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-[#101921] border-[#101921]"
              />
            </div>

            {/* Filters */}
            <div className="space-y-4">
              {/* Role Filters */}
              <div>
                <h3 className="text-sm font-medium mb-3 flex items-center space-x-2">
                  <Filter className="h-4 w-4 text-white" />
                  <span>Filter by Role</span>
                </h3>
                <div className="flex flex-wrap gap-2">
                  {roleFilters.map((filter) => (
                    <Button
                      key={filter.id}
                      variant={selectedRole === filter.id ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedRole(filter.id)}
                      className={cn(
                        selectedRole === filter.id && "shadow-white"
                      )}
                    >
                      <filter.icon className="h-4 w-4 mr-2" />
                      {filter.label}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Status Filters */}
              <div>
                <h3 className="text-sm font-medium mb-3">Filter by Status</h3>
                <div className="flex flex-wrap gap-2">
                  {statusFilters.map((filter) => (
                    <Button
                      key={filter.id}
                      variant={selectedStatus === filter.id ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedStatus(filter.id)}
                      className={cn(
                        selectedStatus === filter.id && "shadow-white"
                      )}
                    >
                      <filter.icon className="h-4 w-4 mr-2" />
                      {filter.label}
                    </Button>
                  ))}
                </div>
              </div>
            </div>

            {/* Results Summary */}
            <div className="flex items-center justify-between pt-4 border-t border-[#101921]">
              <div className="flex items-center space-x-4">
                <span className="text-sm text-muted-foreground">
                  Showing {filteredUsers.length} of {totalCount} personnel
                </span>
                {(searchQuery || selectedRole !== "all" || selectedStatus !== "all") && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setSearchQuery("");
                      setSelectedRole("all");
                      setSelectedStatus("all");
                    }}
                  >
                    Clear Filters
                  </Button>
                )}
              </div>

              <Badge variant="secondary">
                {displayedOnSiteCount} on site
              </Badge>
            </div>
          </div>
        </Card>

        {/* Loading/Error/Personnel Grid */}
        {isLoading ? (
          <Card className="bg-[#1F2733] border-[#424953] p-12 text-center">
            <div className="flex items-center justify-center mb-4">
              <Users className="h-16 w-16 text-white animate-pulse" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Loading Personnel...</h3>
            <p className="text-muted-foreground">Fetching data from the server.</p>
          </Card>
        ) : error ? (
          <Card className="bg-[#1F2733] border-[#424953] p-12 text-center">
            <div className="flex items-center justify-center mb-4">
              <UserX className="h-16 w-16 text-destructive" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Error Loading Personnel</h3>
            <p className="text-destructive-foreground">{error}</p>
            <Button onClick={fetchUsers} className="mt-4" variant="outline">
              Retry Load
            </Button>
          </Card>
        ) : filteredUsers.length === 0 ? (
          <Card className="text-whitebg-[#1F2733] border-[#424953] p-12">
            <div className="text-center">
              <Users className="h-16 w-16 text-white mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Personnel Found</h3>
              <p className="text-muted-foreground">
                {searchQuery || selectedRole !== "all" || selectedStatus !== "all"
                  ? "Try adjusting your search criteria or filters"
                  : "No personnel have been registered yet"
                }
              </p>
            </div>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredUsers.map((user) => (
              <UserCard key={user.id} user={user} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}