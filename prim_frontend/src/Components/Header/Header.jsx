import { Eye } from "lucide-react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";
import { toast } from "react-toastify";

const Header = () => {
  const [loggedInUser, setLoggedInUser] = useState('');
  const navigate = useNavigate();
  const location = useLocation(); // re-renders on route change

  const handleSuccess = (msg) => toast.success(msg);

  const handleLogout = () => {
    localStorage.removeItem("loggedInUser");
    localStorage.removeItem("token");
    handleSuccess('User Logged out');
    navigate('/login');
  };

  useEffect(() => {
    const user = localStorage.getItem("loggedInUser");
    setLoggedInUser(user ? JSON.parse(user).name : '');
  }, [location]); // runs whenever route changes (like after login)

  return (
    <header className="relative z-10 flex justify-between items-center p-6 border-b border-[#263140] bg-[#101921] shadow-[0_2px_10px_rgba(0,0,0,0.5)]">
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

      {loggedInUser ? (
        <div className="flex items-center space-x-4">
          <span className="text-[#F2F2F2]">Hello, <span className="text-[#36D399] font-semibold">{loggedInUser}</span></span>
          <button
            onClick={handleLogout}
            className="border border-[#36D399] text-[#36D399] px-4 py-2 rounded-xl hover:bg-[#36D399] hover:text-[#101921] transition-all duration-300 shadow-[0_0_10px_#36D39966]"
          >
            Logout
          </button>
        </div>
      ) : (
        <Link
          to="/login"
          className="bg-[#101921] border border-[#36D399] text-[#36D399] px-6 py-2 rounded-xl hover:bg-[#36D399] hover:text-[#101921] transition-all duration-300 shadow-[0_0_10px_#36D39966] hover:shadow-[0_0_15px_#36D399AA]"
        >
          Get Started
        </Link>
      )}
    </header>
  );
};

export default Header;
