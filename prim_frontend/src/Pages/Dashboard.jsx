import { useState } from "react";
import Sidebar from "../Components/Sidebar/AppSidebar";
import DashboardView from "../Components/DashboardView/DashboardView";
import Header from "../Components/Header/Header"
import Footer from "../Components/Footer/Footer" 

export default function DashboardLayout() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar collapsed={collapsed} toggleSidebar={() => setCollapsed(!collapsed)} />
        <div className="flex-1 overflow-y-auto bg-[#121821]">
          <DashboardView />
        </div>
      </div>
    </div>
  );
}

