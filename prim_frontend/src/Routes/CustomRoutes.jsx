import { Route, Routes,Navigate } from "react-router-dom";
import { useState } from 'react';
// import RefreshHandler from '../Scripts/refreshhandler';
import Home from "../Pages/Home";
import Dashboard from "../Pages/Dashboard";
import Login from "../Pages/Login";
import Faceid from "../Pages/Faceid";
import Database from "../Pages/Database";


function CustomRoutes() {


    const [isAuthenticated, setIsAuthenticated] = useState(false);

    const PrivateRoute = ({ element }) => {
        return isAuthenticated ? element : <Navigate to="/login" />
    }

    return (
        <div>
            {/* <RefreshHandler setIsAuthenticated={setIsAuthenticated} /> */}
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/home" element={<Home />} />
                <Route path='/dashboard' element={<Dashboard />}/>
                <Route path="/login" element={<Login />} />
                <Route path="/faceid" element={<Faceid />} />
                <Route path="/viewdata" element={<Database />} />
            </Routes>
        </div>
    );
};

export default CustomRoutes;
