import Footer from "../Components/Footer/Footer";
import Header from "../Components/Header/Header";
import LoginTable from "../Components/LoginTable/LoginTable";

function Login(){
    return(
        <div className="min-h-screen bg-[#101B24]">
            <Header/>
            <LoginTable/>
            <Footer/>
        </div>
    )
}

export default Login;