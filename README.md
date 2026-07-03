# 🧠 Facial Recognition System (LBPH + OpenCV)

A modular real-time facial recognition system built with Python and OpenCV.  
This project was developed as part of a Telecommunications Engineering thesis and later refactored into a professional portfolio-grade software project.

---

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-FF6F00?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-4CAF50?style=for-the-badge)
![Biometrics](https://img.shields.io/badge/Biometrics-1E88E5?style=for-the-badge)

---

## 🚀 Project Overview

This system implements a complete facial recognition pipeline:

- Real-time face detection using Haar Cascades
- Dataset creation from webcam input
- Machine Learning model training (LBPH algorithm)
- Real-time face recognition
- Modular and scalable architecture

The goal is to demonstrate a full biometric authentication workflow using classical computer vision techniques.

---

## 🎯 Objectives

- Capture facial images from webcam
- Build a labeled dataset automatically
- Train a facial recognition model (LBPH)
- Perform real-time identity recognition
- Design a clean modular architecture suitable for production-style projects

---

## 🏗️ System Architecture

```text
Capture Faces → Dataset Creation → Model Training → Face Recognition
```

---

## 🧠 Features

- Real-time face detection
- Automated dataset generation via webcam
- LBPH facial recognition model
- Modular architecture (clean separation of concerns)
- Centralized configuration system (`config.py`)
- Label mapping system (no hardcoded identities)
- Cross-platform compatibility
- Interactive CLI menu system

---

## 🛠️ Main Technologies

- Python
- OpenCV
- NumPy
- Haar Cascades (Face Detection)
- LBPH Algorithm
- Computer Vision
- Biometrics

---

## 📂 Project Structure

```bash
facial_project/
│
├── src/
│   ├── main.py             # Entry point (menu system)
│   ├── capture.py          # Face dataset creation
│   ├── train.py            # LBPH model training
│   ├── recognition.py      # Real-time recognition
│   ├── utils.py            # Helper functions
│   ├── config.py           # Global configuration
│   └── Encuadre_de_las_caras.xml
│
├── data/
│   ├── dataset/            # User face images
│   └── models/             # Trained model + labels
│
├── assets/
├── requirements.txt
└── README.md
```
---

## ⚙️ Installation

1. Clone repository
```bash
git clone https://github.com/Marta-Paras-Gutierrez/facial-recognition-access-system.git
cd facial-recognition-access-system
```

2. Create virtual environment (optional)
```bash
python -m venv venv
```
Activate:
```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to run

```bash
python src/main.py
```

---

## 🧭 System Modules

1. Face Capture: Registers new users using webcam and stores facial images.

2. Training Module: Trains LBPH model using dataset images.

3. Recognition Module: Performs real-time face recognition.

---

## 🧠 How It Works

The system uses LBPH (Local Binary Patterns Histograms):

- Converts face images into grayscale patterns
- Extracts local texture features
- Compares histograms for identity matching
- Works efficiently with small datasets and real-time input

---

## 📊 Output Files

After training:
```bash
data/models/
├── lbph_model.xml
└── labels.txt
```
Example:
```bash
0:Marta
1:Carlos
2:Laura
```

---

## 📌 Notes

- Webcam required.
- Haar Cascade file must be present.
- Model must be trained before recognition
- Dataset images are not included for privacy reasons.
- This repository only contains the software implementation.
- Original project included hardware integration and biometric access control experiments.

---
## 🔧 Configuration

All parameters are located in:
```bash
src/config.py
```
You can modify:

- Camera index
- Image size
- Threshold confidence
- Dataset settings

---

## 📌 Future Improvements

- Deep Learning model (CNN-based recognition)
- GUI interface (Tkinter / PyQt)
- REST API version
- Face anti-spoofing detection
- Docker deployment
- Cloud dataset storage

---

## 🎓 Academic Context
- This project is based on my Bachelor's Thesis in Telecommunications Engineering focused on biometric authentication systems.

It has been refactored into a modular software architecture for portfolio presentation purposes.

---

## 📄 License

Educational and portfolio use only.

---

## 👤 Author

Marta Parás
