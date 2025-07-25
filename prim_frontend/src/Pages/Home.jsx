import Features from "../Components/Features/Features";
import Footer from "../Components/Footer/Footer";
import Header from "../Components/Header/Header";
import HeroSection from "../Components/HersSection/HeroSection";

function Home(){
    return(
        <div className="min-h-screen bg-[#101B24]">
            <Header/>
            <HeroSection/>
            <Features/>
            <Footer/>
        </div>
    )
}

export default Home;