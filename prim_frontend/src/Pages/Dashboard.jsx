import DashboardView from "../Components/DashboardView/DashboardView";
import Footer from "../Components/Footer/Footer";
import Header from "../Components/Header/Header";

function Dashboard(){
    return(
        <div className="min-h-screen bg-[#101B24]">
            <Header/>
            <DashboardView/>
            <Footer/>
        </div>
    )
}

export default Dashboard;