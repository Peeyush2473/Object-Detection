# 🔒 Real-time Object Detection for Smart Surveillance

A production-ready, real-time surveillance system that detects suspicious objects and activities from live camera or video feeds using deep learning. Built with **YOLOv8**, **OpenCV**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-purple?logo=ultralytics)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Usage Guide](#-usage-guide)
- [Example Output](#-example-output)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Project Description

Traditional surveillance systems rely on human operators monitoring multiple camera feeds, which is error-prone and expensive. This project leverages state-of-the-art deep learning to **automatically detect suspicious objects** (such as knives, unattended bags, and unauthorized persons) from video feeds in real time.

The system processes each video frame through a **YOLOv8 object detection model**, draws annotated bounding boxes with confidence scores, raises alerts when suspicious objects are detected, and logs all detections to a CSV file for later analysis — all accessible through an interactive **Streamlit dashboard**.

---

## ✨ Features

### Core Features
- **Real-time video processing** from webcam or uploaded video files
- **YOLOv8 object detection** with multiple model size options (nano → extra-large)
- **Bounding boxes** with class labels and confidence scores
- **Alert system** that triggers when suspicious objects are detected
- **Interactive dashboard** with live feed, detection list, and metrics
- **Detection logging** to CSV with timestamps
- **Configurable detection classes** — enable/disable from the sidebar

### Advanced Features
- **Sound alert** when weapons or suspicious objects are detected
- **Email notifications** (optional, configurable)
- **FPS counter** for real-time performance monitoring
- **Detection log viewer** with filtering and CSV export
- **Rate-limited alerts** to prevent notification spam
- **Modular architecture** for easy extension and fine-tuning

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Deep Learning** | PyTorch |
| **Dashboard UI** | Streamlit |
| **Data Processing** | NumPy, Pandas |
| **Dataset** | COCO (80 classes, pretrained) |
| **Language** | Python 3.10+ |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Webcam (for live detection) or a sample video file

### Step-by-step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/object-detection-smart-surveillance.git
   cd object-detection-smart-surveillance
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   # venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "from ultralytics import YOLO; print('YOLOv8 ready ✅')"
   ```

> **Note:** The YOLOv8 model weights (`yolov8n.pt`) are downloaded automatically the first time you run the application. No manual download is required.

---

## ▶️ How to Run

### Launch the Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Quick Start
1. Select an input source: **Webcam** or **Upload Video**
2. Adjust the confidence threshold in the sidebar
3. Enable/disable detection classes as needed
4. Click **▶️ Start Detection**
5. View detections in real time with bounding boxes and alerts
6. Check the **Detection Logs** tab for recorded data

---

## 📘 Usage Guide

### Webcam Mode
1. Select "📷 Webcam" from the input source
2. Click **▶️ Start Detection**
3. The system will use your default camera (device `0`)
4. Click **⏹ Stop** to end the session

### Video Upload Mode
1. Select "📁 Upload Video"
2. Upload an `.mp4`, `.avi`, `.mov`, or `.mkv` file
3. Click **▶️ Start Detection**
4. The system will process the entire video frame-by-frame

### Sidebar Configuration
- **Model variant**: Choose from `yolov8n` (fastest) to `yolov8x` (most accurate)
- **Confidence threshold**: Slide to adjust minimum detection confidence (0.1 – 1.0)
- **Detection classes**: Toggle individual classes on/off
- **Sound alert**: Enable/disable audio alerts for suspicious objects

### Detection Logs
- Navigate to the **📊 Detection Logs** tab
- Filter by class name or alert status
- Download filtered logs as CSV

---

## 📸 Example Output

When running the system, you will see:

```
┌─────────────────────────────────────────────────┐
│  🔒 Smart Surveillance Dashboard                │
│                                                 │
│  ⚡ FPS: 28.3 │ 🖼️ Frames: 142 │ 🚨 Alerts: 1  │
│                                                 │
│  ┌───────────────────┬──────────────────────┐   │
│  │                   │ 🟢 person — 92%      │   │
│  │   [Live Video     │ 🟢 backpack — 87%    │   │
│  │    with bounding  │ 🔴 knife — 73%       │   │
│  │    boxes]         │                      │   │
│  │                   │                      │   │
│  └───────────────────┴──────────────────────┘   │
│                                                 │
│  🚨 ALERT — Suspicious object: knife (14:32:05) │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
objectDetection/
│
├── app/                          # Core application modules
│   ├── __init__.py
│   ├── detection.py              # YOLOv8 wrapper & inference
│   ├── video_processor.py        # Frame processing pipeline
│   └── utils.py                  # Logging, alerts, helpers
│
├── dashboard/
│   └── streamlit_app.py          # Streamlit dashboard UI
│
├── models/                       # YOLO weights (auto-downloaded)
│   └── .gitkeep
│
├── data/                         # Sample videos & detection logs
│   └── .gitkeep
│
├── notebooks/                    # Jupyter notebooks for training
│   └── .gitkeep
│
├── config.py                     # Central configuration
├── requirements.txt              # Python dependencies
├── report.md                     # Detailed project report
├── .gitignore
└── README.md                     # This file
```

---

## ⚙️ Configuration

All settings are centralized in [`config.py`](config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `yolov8n.pt` | YOLO model variant |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum confidence for detections |
| `IOU_THRESHOLD` | `0.50` | IoU threshold for NMS |
| `MONITORED_CLASSES` | person, knife, scissors, backpack, ... | Classes to detect |
| `ALERT_CLASSES` | knife | Classes that trigger alerts |
| `ENABLE_SOUND_ALERT` | `False` | Enable audio alert |
| `ENABLE_EMAIL_ALERT` | `False` | Enable email notifications |
| `MAX_DISPLAY_WIDTH` | `720` | Max frame width for display |

### Email Alerts (Optional)
To enable email notifications, edit `config.py`:
```python
ENABLE_EMAIL_ALERT = True
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver@example.com"
```
> Use a Gmail [App Password](https://support.google.com/accounts/answer/185833), not your regular password.

---

## 🔮 Future Improvements

- 🎥 **Multi-camera support** — monitor multiple feeds simultaneously
- 👤 **Facial recognition** — identify known vs unknown individuals
- 🧠 **Anomaly detection** — detect unusual motion patterns using LSTM/autoencoder
- 📱 **Mobile app** — push notifications to mobile devices
- ☁️ **Cloud deployment** — deploy on AWS/GCP with GPU instances
- 🎯 **Custom model training** — fine-tune on weapon-specific datasets
- 📊 **Analytics dashboard** — historical trend analysis and heatmaps
- 🗄️ **Database storage** — replace CSV with PostgreSQL/MongoDB

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Python, YOLOv8, and Streamlit
</p>