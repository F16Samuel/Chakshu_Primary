import { Link } from "react-router-dom";

const Footer = () =>{
    return(
     <footer className="relative z-10 border-t border-[#1F2733] p-6 text-center">
            <div className="max-w-4xl mx-auto">
            <p className="text-[#919293] mb-4">
                Chakshu - The Eye That Thinks Before You Blink
            </p>
            <div className="flex justify-center space-x-8">
                <Link to='https://github.com/F16Samuel/Chakshu_Primary' className="text-[#36D399] hover:text-[#00AAFF] transition-colors">GitHub</Link>
                <Link to='/dashboard' className="text-[#36D399] hover:text-[#00AAFF] transition-colors">View Demo</Link>
                <Link to='/' className="text-[#36D399] hover:text-[#00AAFF] transition-colors">Home</Link>
                <Link to='/login' className="text-[#36D399] hover:text-[#00AAFF] transition-colors">Admin Login</Link>
            </div>
            </div>
     </footer>
    )
}

export default Footer;
