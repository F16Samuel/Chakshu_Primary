import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";
import axios from "axios";

const LoginTable = () => {
  const [mode, setMode] = useState(null); // 'login' or 'register'
  const [formData, setFormData] = useState({}); // Stores form input data
  const navigate = useNavigate(); // Hook for navigation

  // Define questions for login and registration forms
  const questions = {
    login: [
      { name: "email", label: "Email", placeholder: "Enter your email" },
      { name: "password", label: "Password", placeholder: "Enter your password" },
    ],
    register: [
      { name: "name", label: "Name", placeholder: "Enter your name" },
      { name: "email", label: "Email", placeholder: "Enter your email" },
      { name: "password", label: "Password", placeholder: "Create a password" },
      { name: "confirmPassword", label: "Confirm Password", placeholder: "Repeat your password" },
    ],
  };

  // Handles changes in form input fields
  const handleChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value, // Update the corresponding form data field
    }));
  };

  // Handles form submission for login and registration
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent default form submission behavior

    // Handle login
    if (mode === "login") {
      try {
        // Make a POST request to the login API endpoint using the environment variable
        const res = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/auth/login`, {
          email: formData.email,
          password: formData.password,
        });

        // Store JWT token and user info in local storage
        localStorage.setItem("token", res.data.jwtToken);
        localStorage.setItem("loggedInUser", JSON.stringify({ name: res.data.name }));
        toast.success("Login successful!"); // Show success toast
        navigate("/dashboard"); // Navigate to dashboard on success
      } catch (err) {
        // Show error toast if login fails
        toast.error(err.response?.data?.message || "Login failed.");
      }
    }

    // Handle registration
    if (mode === "register") {
      // Check if passwords match
      if (formData.password !== formData.confirmPassword) {
        toast.error("Passwords do not match.");
        return;
      }

      try {
        // Make a POST request to the signup API endpoint using the environment variable
        const res = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/auth/signup`, {
          name: formData.name,
          email: formData.email,
          password: formData.password,
        });
        toast.success("Registration successful!"); // Show success toast
        navigate("/home"); // Navigate to home on success
      } catch (err) {
        // Show error toast if registration fails
        toast.error(err.response?.data?.message || "Registration failed.");
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#101B24] flex items-center justify-center px-4">
      <div className="bg-[#171D26] border border-[#252F3D] rounded-2xl shadow-lg p-10 w-full max-w-md text-white">
        <h2 className="text-3xl font-bold text-center mb-6">Welcome to Chakshu</h2>

        {!mode ? (
          // Initial view with Login and Register buttons
          <div className="flex flex-col space-y-4">
            <button
              onClick={() => setMode("login")}
              className="bg-gradient-to-r from-[#36D399] to-[#00AAFF] text-[#101921] px-6 py-3 rounded-xl font-semibold hover:shadow-[0_0_12px_#36D399] transition-all duration-300"
            >
              Login
            </button>
            <button
              onClick={() => setMode("register")}
              className="border border-[#00AAFF] text-[#00AAFF] px-6 py-3 rounded-xl font-semibold hover:bg-[#00AAFF] hover:text-[#101921] transition-all duration-200"
            >
              Register
            </button>
          </div>
        ) : (
          // Login/Register form
          <form className="space-y-5" onSubmit={handleSubmit}>
            <h3 className="text-2xl font-semibold text-center mb-4 capitalize">{mode} Details</h3>

            {questions[mode].map((q, idx) => (
              <div key={idx}>
                <label className="block mb-1 text-[#C0C0C0]">{q.label}</label>
                <input
                  type={q.name.toLowerCase().includes("password") ? "password" : "text"}
                  name={q.name}
                  value={formData[q.name] || ""}
                  onChange={handleChange}
                  placeholder={q.placeholder}
                  required
                  className="w-full px-4 py-2 rounded-lg bg-[#1F2733] text-white border border-[#252F3D] focus:outline-none focus:border-[#36D399] transition-all"
                />
              </div>
            ))}

            <button
              type="submit"
              className="w-full bg-gradient-to-r from-[#36D399] to-[#00AAFF] text-[#101921] px-6 py-3 rounded-xl font-semibold hover:shadow-[0_0_12px_#36D399] transition-all duration-300"
            >
              Submit
            </button>

            <button
              type="button"
              onClick={() => setMode(null)}
              className="w-full mt-3 text-sm text-[#C0C0C0] hover:underline text-center"
            >
              ← Go back
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default LoginTable;
