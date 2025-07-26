import { Camera, Monitor, Activity, Plus } from "lucide-react";
import { Link } from "react-router-dom";

const DashboardView = () => {
    return(
    <section className="relative z-10 px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-4xl text-white font-bold text-center mb-16">
            Real-time <span className="text-[#36D399]">Surveillance</span> Dashboard
          </h3>
          
          <div className="bg-[#171D26] border border-[#252F3D] rounded-xl overflow-hidden shadow-elevated">
            
            
            {/* Dashboard Header */}
            <div className="bg-[#1F2733] p-4 border-b border-[#252F3D]">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-[#36D399] rounded-full animate-pulse"></div>
                    <span className="text-white text-sm font-medium">SYSTEM ACTIVE</span>
                  </div>
                  <div className="text-sm text-[#888A8C]">
                    Last Update: {new Date().toLocaleTimeString()}
                  </div>
                </div>
                <div className="flex space-x-2">
                  <div className="px-3 py-1 bg-[#36D399] text-primary-foreground rounded text-xs font-bold">
                    4 CAMERAS ONLINE
                  </div>
                  <div className="px-3 py-1 bg-[#00AAFF] text-primary-foreground rounded text-xs font-bold">
                    0 THREATS
                  </div>
                </div>
              </div>
            </div>


            
            {/* Dashboard Content */}


            <div className="p-6">
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Camera Feeds */}
                <div className="lg:col-span-2">
                  <h4 className="text-lg text-white font-semibold mb-4 flex items-center">
                    <Camera className="mr-2 h-5 w-5 text-[#36D399]" />
                    Live Camera Feeds
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    {[1, 2, 3, 4].map((i) => (
                      <div key={i} className="bg-[#1F2733] rounded-lg p-4 relative overflow-hidden">
                        <div className="absolute top-2 left-2 z-10">
                          <span className="bg-[#36D399] text-primary-foreground px-2 py-1 rounded text-xs font-bold">
                            CAM {i}
                          </span>
                        </div>
                        <div className="aspect-video bg-gradient-to-r from-[#21383F] to-[#1F3642] rounded border border-transparent flex items-center justify-center">
                          <Monitor className="h-8 w-8 text-[#A6A6A6]" />
                        </div>
                        <div className="mt-2 text-sm text-[#9D9E9E] flex justify-center">
                          {i === 1 && "No Threat"}
                          {i === 2 && "No Threat"}
                          {i === 3 && "No Threat"}
                          {i === 4 && "No Threat"}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>


                
                {/* Status Panel */}
                <div>
                  <h4 className="text-white text-lg font-semibold mb-4 flex items-center">
                    <Activity className="mr-2 h-5 w-5 text-[#00AAFF]" />
                    System Status
                  </h4>
                  <div className="space-y-4">
                    <div className="bg-[#1F2733] rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white text-sm font-medium">Fight Detection</span>
                        <span className="text-xs text-[#36D399]">ACTIVE</span>
                      </div>
                      <div className="w-full bg-background rounded-full h-2">
                        <div className="bg-[#36D399] h-2 rounded-full w-full"></div>
                      </div>
                    </div>
                    
                    <div className="bg-[#1F2733] rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white text-sm font-medium">Weapon Scan</span>
                        <span className="text-xs text-[#CC66FF]">ACTIVE</span>
                      </div>
                      <div className="w-full bg-background rounded-full h-2">
                        <div className="bg-[#CC66FF] h-2 rounded-full w-full"></div>
                      </div>
                    </div>
                    
                    <div className="bg-[#1F2733] rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white text-sm font-medium">Face ID System</span>
                        <span className="text-xs text-[#00AAFF]">ACTIVE</span>
                      </div>
                      <div className="w-full bg-background rounded-full h-2">
                        <div className="bg-[#00AAFF] h-2 rounded-full w-full"></div>
                      </div>
                    </div>
                    
                    <div className="bg-[#1F2733] rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white text-sm font-medium">Zone Monitor</span>
                        <span className="text-xs text-[#FF9933]">ACTIVE</span>
                      </div>
                      <div className="w-full bg-background rounded-full h-2">
                        <div className="bg-[#FF9933] h-2 rounded-full w-full"></div>
                      </div>
                    </div>

                    </div>
                </div>
              </div>
            </div>
          </div>
        </div>
    </section>
    )
}

export default DashboardView;