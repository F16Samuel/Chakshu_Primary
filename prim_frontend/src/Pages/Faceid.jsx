import Faceidc from "../Components/Faceidc/Faceidc";
import Footer from "../Components/Footer/Footer";
import Header from "../Components/Header/Header";

function Faceid(){
    return(
        <div className="min-h-screen bg-[#101B24]">
            <Header/>
            <Faceidc/>
            <Footer/>
        </div>
    )
}

export default Faceid;