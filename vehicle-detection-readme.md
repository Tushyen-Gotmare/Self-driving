# 🚗 Advanced Vehicle Detection System

## Overview

This is an intelligent computer vision-based vehicle detection and tracking system built using Streamlit, OpenCV, and YOLOv8. The application provides real-time analysis of video streams, offering comprehensive insights into vehicle movements, traffic signals, and potential hazards.

## 🌟 Key Features

- **Real-time Object Detection**
  - Detect multiple vehicle types (Cars, Motorcycles, Buses, Trucks)
  - Identify traffic signals and their colors
  - Track lane markings and suggest driving directions

- **Proximity Warnings**
  - Automatic status indicators: GO, SLOW, STOP
  - Dynamic color-coded warnings based on vehicle distance
  - Configurable detection thresholds

- **Additional Capabilities**
  - License plate text recognition
  - Lane direction estimation
  - Intuitive Streamlit-based user interface

## 🛠 Technologies Used

- Python
- Streamlit
- OpenCV
- YOLOv8
- Pytesseract
- NumPy

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-detection-system.git
cd vehicle-detection-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
- **MacOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## 🚀 Usage

Run the application:
```bash
streamlit run app.py
```

### How to Use

1. Upload a video file (MP4, AVI, MOV)
2. The system will automatically analyze the video
3. Use sidebar controls to adjust detection sensitivity

## 📊 Detection Modes

- **GO (Green)**: No immediate threats
- **SLOW (Yellow)**: Vehicles at moderate distance
- **STOP (Red)**: Vehicles too close or red traffic signal

## 🛡 Customization

Adjust detection parameters in the sidebar:
- Stop Threshold
- Slow Threshold
- Detection Confidence

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

## 📧 Contact

Tushyen Gotmare - [tushyengotmare.com]


