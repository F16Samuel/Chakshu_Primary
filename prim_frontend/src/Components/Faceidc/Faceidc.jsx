import { useRef, useState } from "react";
import Webcam from "react-webcam";

const Faceidc = () => {
  const webcamRef = useRef(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [submitted, setSubmitted] = useState(false);

  // Additional form fields
  const [name, setName] = useState("");
  const [regId, setRegId] = useState("");
  const [status, setStatus] = useState("Student");

  const capture = () => {
    if (!name || !regId || !status) {
      alert("Please fill in all the details before capturing your face.");
      return;
    }
    const image = webcamRef.current.getScreenshot();
    setImageSrc(image);
  };

  const submitFace = () => {
    // TODO: Send data to backend API
    const payload = {
      name,
      regId,
      status,
      image: imageSrc,
    };
    console.log("Submitting face data:", payload);
    setSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-[#101B24] flex flex-col items-center justify-center p-6 text-white">
      <h2 className="text-3xl font-bold mb-6">Register Face ID</h2>

      {!submitted ? (
        <>
          <div className="bg-[#171D26] border border-[#252F3D] rounded-xl p-6 w-full max-w-md shadow-lg">
            <div className="mb-4">
              <label className="block mb-2 text-sm font-semibold text-[#C0C0C0]">Full Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full p-3 rounded-lg bg-[#1F2733] text-white border border-[#252F3D] focus:outline-none focus:ring-2 focus:ring-[#36D399]"
                placeholder="Enter your full name"
              />
            </div>

            <div className="mb-4">
              <label className="block mb-2 text-sm font-semibold text-[#C0C0C0]">Registration No / Staff ID</label>
              <input
                type="text"
                value={regId}
                onChange={(e) => setRegId(e.target.value)}
                className="w-full p-3 rounded-lg bg-[#1F2733] text-white border border-[#252F3D] focus:outline-none focus:ring-2 focus:ring-[#36D399]"
                placeholder="Enter your ID"
              />
            </div>

            <div className="mb-6">
              <label className="block mb-2 text-sm font-semibold text-[#C0C0C0]">Status</label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="w-full p-3 rounded-lg bg-[#1F2733] text-white border border-[#252F3D] focus:outline-none focus:ring-2 focus:ring-[#36D399]"
              >
                <option value="Student">Student</option>
                <option value="Professor">Professor</option>
                <option value="Staff">Staff</option>
              </select>
            </div>

            {!imageSrc ? (
              <>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="rounded-lg shadow-lg border-4 border-[#252F3D] w-full"
                />
                <button
                  onClick={capture}
                  className="mt-5 w-full bg-gradient-to-r from-[#36D399] to-[#00AAFF] text-[#101921] px-6 py-3 rounded-xl font-semibold hover:shadow-[0_0_12px_#36D399] transition-all duration-300"
                >
                  Capture Face
                </button>
              </>
            ) : (
              <>
                <img src={imageSrc} alt="Captured face" className="rounded-lg border border-[#252F3D] shadow-lg w-full mt-3" />
                <div className="flex gap-4 mt-4">
                  <button
                    onClick={() => setImageSrc(null)}
                    className="border border-[#00AAFF] text-[#00AAFF] px-6 py-2 rounded-lg hover:bg-[#00AAFF] hover:text-[#101921] transition w-1/2"
                  >
                    Retake
                  </button>
                  <button
                    onClick={submitFace}
                    className="bg-gradient-to-r from-[#36D399] to-[#00AAFF] text-[#101921] px-6 py-2 rounded-lg hover:shadow-[0_0_12px_#36D399] transition w-1/2"
                  >
                    Submit
                  </button>
                </div>
              </>
            )}
          </div>
        </>
      ) : (
        <div className="text-center">
          <h3 className="text-2xl font-semibold mt-4 text-green-400">✅ Face ID Registered Successfully!</h3>
        </div>
      )}
    </div>
  );
};

export default Faceidc;
