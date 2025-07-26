import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "../src/components/layout/AppSidebar";
import { Header } from "@/components/layout/Header";
import Dashboard from "./pages/Dashboard";
import Register from "./pages/Register";
import Personnel from "./pages/Personnel";
import Monitor from "./pages/Monitor";
import Manual from "./pages/Manual";
import NotFound from "./pages/NotFound";
import { useState } from "react";

const queryClient = new QueryClient();
const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <SidebarProvider>
          <div className="min-h-screen flex w-full bg-gradient-primary">
            <AppSidebar />
            <div className="flex-1 flex flex-col">
              <main className="flex-1">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/register" element={<Register />} />
                  <Route path="/personnel" element={<Personnel />} />
                  <Route path="/monitor" element={<Monitor />} />
                  <Route path="/manual" element={<Manual />} />
                  {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </main>
            </div>
          </div>
        </SidebarProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
