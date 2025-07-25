import {DoorClosedLocked, AlertTriangle, UserCheck, Activity} from "lucide-react";

const Features = () => {
    return(
              <section className="relative z-10 px-6 py-20">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-4xl font-bold text-center mb-16 text-white">
            Advanced <span className="text-[#00AAFF]">Detection</span> Capabilities
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Fight Detection */}
            <div className="group bg-[#171D26] border border-[#252F3D] rounded-2xl p-8 hover:border-[#36D399] transition-all duration-300 hover:shadow-[0_0_20px_4px_#36D399]/50">
              <div className="flex items-center mb-6">
                <div className="bg-[#1F2733] p-3 text-white rounded-lg mr-4 group-hover:bg-[#36D399] group-hover:text-[#101921] transition-all duration-300">
                  <Activity className="h-8 w-8" />
                </div>
                <h4 className="text-white text-2xl font-bold">Fight Detection</h4>
              </div>
              <p className="text-[#919293] mb-4">
                Real-time pose estimation and motion analysis to instantly detect physical altercations between students.
              </p>
              <ul className="text-white space-y-2 text-sm">
                <li className="flex items-center">
                  <div className=" w-2 h-2 bg-[#36D399] rounded-full mr-3"></div>
                  Advanced pose estimation AI
                </li>
                <li className="flex items-center">
                  <div className=" w-2 h-2 bg-[#36D399] rounded-full mr-3"></div>
                  Motion pattern analysis
                </li>
                <li className="flex items-center">
                  <div className=" w-2 h-2 bg-[#36D399] rounded-full mr-3"></div>
                  Instant alert system
                </li>
              </ul>
            </div>

            {/* Weapon Detection */}
            <div className="group bg-[#171D26] border border-[#252F3D] rounded-xl p-8 hover:border-[#FF9933] transition-all duration-300 hover:shadow-[0_0_20px_4px_#FF9933]/50">
              <div className="flex items-center mb-6">
                <div className="bg-[#1F2733] p-3 rounded-lg mr-4 text-white group-hover:bg-[#FF9933] group-hover:text-primary-foreground transition-all duration-300">
                  <AlertTriangle className="h-8 w-8" />
                </div>
                <h4 className="text-white text-2xl font-bold">Weapon & Bomb Detection</h4>
              </div>
              <p className="text-[#919293] mb-4">
                Identifies dangerous objects including knives, firearms, and suspicious packages using advanced object detection.
              </p>
              <ul className="text-white space-y-2 text-sm">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#FF9933] rounded-full mr-3"></div>
                  Multi-object recognition
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#FF9933]  rounded-full mr-3"></div>
                  Threat level assessment
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#FF9933]  rounded-full mr-3"></div>
                  Emergency response protocols
                </li>
              </ul>
            </div>

            {/* Face ID */}
            <div className="group bg-[#171D26] border border-[#252F3D] rounded-xl p-8 hover:border-[#00AAFF] transition-all duration-300 hover:shadow-[0_0_20px_4px_#00AAFF]/50">
              <div className="flex items-center mb-6">
                <div className="bg-[#1F2733] text-white p-3 rounded-lg mr-4 group-hover:bg-[#00AAFF] group-hover:text-primary-foreground transition-all duration-300">
                  <UserCheck className="h-8 w-8" />
                </div>
                <h4 className="text-white text-2xl font-bold">Face ID Campus Entry</h4>
              </div>
              <p className="text-[#919293] mb-4">
                Seamless campus access through facial recognition technology, eliminating the need for physical ID cards.
              </p>
              <ul className="text-white space-y-2 text-sm">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#00AAFF] rounded-full mr-3"></div>
                  Contactless entry system
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#00AAFF] rounded-full mr-3"></div>
                  Student database integration
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#00AAFF] rounded-full mr-3"></div>
                  Access history tracking
                </li>
              </ul>
            </div>

            {/* Restricted Areas */}
            <div className="group bg-[#171D26] border border-[#252F3D] rounded-xl p-8 hover:border-[#CC66FF] transition-all duration-300 hover:shadow-[0_0_20px_4px_#CC66FF]/50">
              <div className="flex items-center mb-6">
                <div className="bg-[#1F2733] text-white p-3 rounded-lg mr-4 group-hover:bg-[#CC66FF] group-hover:text-primary-foreground transition-all duration-300">
                  <DoorClosedLocked className="h-8 w-8" />
                </div>
                <h4 className="text-white text-2xl font-bold">Restricted Area Alerts</h4>
              </div>
              <p className="text-[#919293] mb-4">
                Monitors unauthorized access to sensitive areas like laboratories, server rooms, and rooftops.
              </p>
              <ul className="text-white space-y-2 text-sm">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#CC66FF] rounded-full mr-3"></div>
                  Zone-based monitoring
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#CC66FF] rounded-full mr-3"></div>
                  Permission verification
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-[#CC66FF] rounded-full mr-3"></div>
                  Real-time notifications
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    )
}

export default Features;