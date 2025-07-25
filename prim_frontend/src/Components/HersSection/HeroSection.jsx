import { Zap, Activity } from "lucide-react";

const HeroSection = () => {
  return (
    <>
      <section className="relative z-10 px-6 py-20 text-center bg-[#101B24] text-white">
        <div className="max-w-4xl mx-auto">
          {/* Heading */}
          <div className="mb-12">
            <h2 className="text-5xl sm:text-6xl font-bold mb-6 animate-glow">
              AI-Powered Campus
              <span className="block text-[#36D399]">Security System</span>
            </h2>
            <p className="text-lg sm:text-xl text-[#C0C0C0] mb-8 max-w-2xl mx-auto">
              Advanced surveillance technology that monitors, detects, and responds to threats in real-time. 
              Protecting your campus with the power of artificial intelligence.
            </p>
          </div>

          {/* Buttons */}
          <div className="flex justify-center flex-wrap gap-6 mb-12">
            <button className="bg-gradient-to-r from-[#36D399] to-[#00AAFF] text-[#101921] animate-float px-8 py-4 rounded-lg text-lg font-semibold hover:shadow-[0_0_12px_#36D399] transition-all duration-300">
              <Zap className="inline-block mr-2 h-5 w-5" />
              Start Monitoring
            </button>

            <button className="border border-[#00AAFF] text-[#00AAFF] px-8 py-4 rounded-lg text-lg font-semibold hover:bg-[#00AAFF] hover:text-[#101921] transition-all duration-300">
              <Activity className="inline-block mr-2 h-5 w-5" />
              View Demo
            </button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 max-w-2xl mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-[#36D399] mb-2">99.8%</div>
              <div className="text-muted-foreground">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-[#00AAFF] mb-2">&lt;2s</div>
              <div className="text-muted-foreground">Response Time</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-[#CC66FF] mb-2">24/7</div>
              <div className="text-muted-foreground">Monitoring</div>
            </div>
          </div>
        </div>
      </section>

      <style jsx>{`
        @keyframes glow {
          0%, 100% {
            text-shadow: 0 0 5px #36D399;
          }
          50% {
            text-shadow: 0 0 20px #36D399, 0 0 30px #00AAFF;
          }
        }

        .animate-glow {
          animation: glow 2s ease-in-out infinite;
        }

        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-10px);
          }
        }

        .animate-float {
          animation: float 0.3s ease-in-out infinite;
        }
      `}</style>
    </>
  );
};

export default HeroSection;
