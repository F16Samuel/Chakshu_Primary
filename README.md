# 👁️‍🗨️ Chakshu – AI-Powered Campus Security System

## 🔍 Problem Statement

In large campus environments, ensuring student safety and managing real-time security threats remains a significant challenge. Manual monitoring of CCTV feeds is inefficient and prone to human error. There is an urgent need for an intelligent, automated surveillance system that can recognize faces, detect violent or suspicious activity, and instantly notify authorities—minimizing response time and preventing escalation.

---

## 🧠 Overview

**Chakshu** is an AI-powered campus security platform designed to monitor, detect, and respond to safety threats in real time. It combines facial recognition, object detection, and activity monitoring with clean dashboards for security personnel and admins. With Chakshu, institutions can ensure safer campuses through automation, visibility, and faster responses.

---

## 🚀 Features

- 🧑‍💼 **Face Recognition & Tracking**  
  Identifies and monitors individuals across campus using face ID to flag unknown or blacklisted persons.

- 🗡️ **Weapon Detection**  
  AI scans CCTV streams for weapons like knives or firearms, triggering alerts for immediate intervention.

- 🤼 **Fight & Violence Detection**  
  Detects signs of physical altercations or aggressive movement, using motion analysis and posture recognition.

- 🚫 **Restricted Zone Surveillance**  
  Flags unauthorized access to high-security or no-entry zones using geofencing and object/person detection.

- 🎛️ **Dual Dashboard Access**  
  - **Security Officers:** Live video alerts, incident log access, and manual override tools.  
  - **Admins:** Manage user database, view event history, update security zones, and train the system with new inputs.

---

## 🧪 Functionality Overview

- 🔴 Real-time CCTV feed analysis with visual alerts  
- 🧠 ML inference using YOLOv8 + OpenCV-based pipelines  
- 🪪 Facial ID tagging for registered vs unrecognized individuals  
- 📩 Backend alert system for fights and weapons
- 🧩 Modular design for easy addition of new detection models  
- 🧑‍💻 Dashboard access control

---

## 🛠️ Tech Stack

| Layer        | Technology             |
|--------------|------------------------|
| Frontend     | React.js, TailwindCSS  |
| Backend      | Node.js, FastAPI, Express.js |
| ML Model     | YOLOv8, OpenCV, YOLOv11         |
| Database     | MongoDB, SQLite       |

---

## 🔮 Future Scope
  
- 🧑‍🤝‍🧑 **Crowd Density Detection** – Detect overcrowding during events or emergencies for better crowd control.  
- 🛜 **IoT Integration** – Direct activation of locks, or emergency lights based on threat detection.
- 🎙️ **Voice-Based Alert Commands** – Security personnel can use voice commands to Alert the nearest guard to the threat location.    
- 🛰️ **Integration with Campus Maps** – Real-time marking of threat zones on a visual map interface.

---

## ⚠️ Limitations

- 🧩 **Limited Edge Compatibility** – High-performance models may lag on low-resource systems without GPU acceleration.  
- 🏷️ **Face Recognition Errors** – Lighting, occlusion, or camera angles may reduce recognition accuracy.  
- 📡 **Requires Stable Video Input** – Disruptions in CCTV feeds can delay detection or raise false alarms.

---

## 🔗 Useful Links

- 🚀 **Live Demo**: [chakshu-secure.vercel.app](https://chakshu-primary.vercel.app/)  
- 📂 **Project Repository**: [GitHub – Chakshu](https://github.com/F16Samuel/Chakshu_Primary)  
- 📄 **ML Models**: [Drive Link](https://drive.google.com/drive/folders/1w9wKDXRXCGIU5BPYI2UNxp0knzGegbg1?usp=sharing)  
- 👁️‍🗨️ **Figma Design**: [Figma](https://www.figma.com/design/LMMabVQGV3inSBGkcJ6aLc/Chakshu-Design?node-id=0-1&t=Fsva6wZEF3RkJuqT-1)

---

> 🛡️ Chakshu - eye that thinks before you blink.  
