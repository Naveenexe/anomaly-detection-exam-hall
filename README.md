# 🎯 Anomaly Detection in Exam Hall

## 🚀 **Enhanced Version 2.1**

An advanced AI-powered academic integrity monitoring system that uses state-of-the-art computer vision and deep learning to analyze classroom videos and detect suspicious behaviors during examinations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1+-red.svg)
![Status](https://img.shields.io/badge/status-production--ready-success.svg)

### ✨ **What's New in Version 2.1**
- 🎨 **Complete UI Overhaul** - Modern, responsive design with gradient themes
- 🛡️ **Enhanced Error Handling** - Comprehensive error management and user feedback
- ⚙️ **Configuration System** - External config file for easy parameter tuning
- 📊 **Advanced Analytics** - Interactive dashboards with detailed visualizations
- 🔧 **Improved Performance** - Memory optimization and faster processing
- 📝 **Professional Logging** - Complete logging system for debugging
- ✅ **Input Validation** - Robust validation for all user inputs
- 🎯 **Better Accuracy** - Improved model architecture and training
 - 📁 **Reorganized Structure** - Introduced `src/`, `models/`, `outputs/`, `logs/`, `config/`
 - 🧩 **Modular Imports** - All core modules now imported via `src.*`
 - 🧪 **Best Model Preference** - Auto-loads `models/best_classroom_model.h5` if present
 - 🔐 **Authentication & Audit** - Role-based access (admin / proctor) + JSON audit trail
 - 🖼️ **Snapshot Archiving** - Older snapshots auto-moved to `history/` before new run
 - 📄 **PDF & CSV Export** - One-click report generation with charts & anomaly samples
 - 🗑️ **Privacy Blur Removed** - Only pixelation / black-box retained (UI privacy page retired)
 - 📊 **Distribution Logic Refined** - Zero-count classes shown in bar chart, pie limited to detected behaviors

## 🚀 Features

### 🤖 **AI & Machine Learning**
- **Advanced CNN Architecture**: 4-layer convolutional neural network with batch normalization
- **9 Behavior Classes**: Comprehensive coverage of academic misconduct patterns
- **High Accuracy**: 77%+ validation accuracy with continuous improvement
- **Real-time Processing**: Optimized for live video analysis with frame skipping
- **Confidence Scoring**: Detailed confidence metrics for each detection

### 🎨 **Enhanced User Interface**
- **Modern Design**: Gradient-based UI with professional styling
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Interactive Dashboards**: Advanced charts and visualizations using Plotly
- **Real-time Metrics**: Live status indicators and performance monitoring
- **Intuitive Navigation**: Clean sidebar navigation with contextual help

### 🚨 **Instant Alert System**
- **Real-time Notifications**: Immediate alerts when anomalies are detected
- **Detailed Anomaly Reports**: Comprehensive analysis with behavior breakdown
- **Visual Dashboard Alerts**: Interactive alerts with timeline visualization
- **Snapshot Integration**: Automatic capture of suspicious activities

### 📊 **Analytics & Reporting**
- **Comprehensive Dashboards**: Real-time monitoring with key performance indicators
- **Anomaly Timeline**: Interactive timeline visualization of detected behaviors
- **Behavior Distribution**: Pie charts and statistical analysis
- **Snapshot Gallery**: Automatic capture and organized display of anomalies
- **Video Analysis**: Batch processing with detailed reporting
 - **Data Exports**: PDF (ReportLab) + CSV + Audit Log export
 - **Model Info**: Active model path & class list (optional small panel)

### 🛡️ **Reliability & Performance**
- **Error Handling**: Comprehensive error management with user-friendly messages
- **Input Validation**: Robust validation for all user inputs and file uploads
- **Memory Optimization**: Efficient memory usage preventing leaks
- **Configuration System**: External config file for easy parameter tuning
- **Logging System**: Complete logging for debugging and monitoring

### 🔧 **Technical Excellence**
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Configuration Management**: JSON-based configuration system
- **Performance Monitoring**: Built-in performance tracking and optimization
- **Cross-platform**: Windows, Linux, and macOS compatibility
- **GPU Support**: CUDA-enabled for faster training and inference

### 🔐 **Authentication & Roles**
- Secure login (default users: `admin` / `admin123`, `proctor` / `proctor123`)
- Role-based pages (Model Training / User Management / Audit Log restricted to admin)
- Audit events persisted to `logs/audit_log.json`

### 📄 **Reporting & Exports**
- PDF report (summary table, detailed log, snapshot grid)
- CSV export with metadata header rows
- Audit log CSV export
- All artifacts saved under `outputs/`

## 🎯 Detected Behaviors

1. **Normal** - Regular exam behavior
2. **Discussing** - Students having conversations
3. **Peeking** - Looking at others' work
4. **Cheat Passing** - Passing answers between students
5. **Copying** - Copying from others' papers
6. **Showing Answer** - Displaying answers to others
7. **Suspicious** - General suspicious activities
8. **Using Copy Cheat** - Using unauthorized materials
9. **Using Mobile** - Mobile phone usage during exam

## 📸 Screenshots

### Dashboard
![Dashboard Overview](screenshots/Screenshot%20(72).png)
*Real-time monitoring dashboard with KPIs and quick actions*

### Login & Authentication
![Login Screen](screenshots/Screenshot%20(73).png)
*Secure authentication with role-based access control*

### Video Analysis
![Video Analysis](screenshots/Screenshot%20(74).png)
*Upload and analyze exam hall videos with adjustable parameters*

### Anomaly Detection Results
![Detection Results](screenshots/Screenshot%20(75).png)
*Interactive timeline showing detected anomalies with confidence scores*

### Behavior Distribution
![Behavior Distribution](screenshots/Screenshot%20(76).png)
*Visual analytics with pie charts and bar graphs*

### Snapshot Gallery
![Snapshot Gallery](screenshots/Screenshot%20(77).png)
*Automatic capture and display of anomalous behaviors*

### Reports & Export
![Reports Page](screenshots/Screenshot%20(78).png)
*Generate PDF and CSV reports with one click*

### Model Training
![Model Training](screenshots/Screenshot%20(79).png)
*Admin interface for training new models*

### User Management
![User Management](screenshots/Screenshot%20(80).png)
*Admin panel for managing users and roles*

### Audit Log
![Audit Log](screenshots/Screenshot%20(81).png)
*Complete audit trail of system events and user actions*

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras, CNN
- **Computer Vision**: OpenCV, Image Processing
- **Web Framework**: Streamlit
- **Data Visualization**: Plotly
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- GPU recommended for faster processing
- Video files for analysis (MP4, AVI, MOV, MKV supported)

## ⚡ Quick Start (Basic App)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naveenexe/anomaly-detection-exam-hall.git
   cd anomaly-detection-exam-hall
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the basic application**
  ```bash
  streamlit run streamlit_app.py
  ```

4. **Run the enhanced application (with auth, reports, dashboard)**
  ```bash
  streamlit run streamlit_app_enhanced.py
  ```

4. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

## 📁 Current Project Structure

```
├── streamlit_app.py                # Basic Streamlit interface
├── streamlit_app_enhanced.py       # Auth + dashboard + reporting
├── src/                           # Core package
│   ├── video_analyzer.py          # Video processing & anomaly detection
│   ├── model_trainer.py           # CNN training logic
│   ├── report_generator.py        # PDF/CSV/audit export logic
│   ├── auth_manager.py            # Authentication & audit logging
│   └── privacy_enhancer.py        # (Legacy) pixelate / black-box
├── models/                        # Model artifacts
│   ├── best_classroom_model.h5    # Preferred checkpoint (if exists)
│   ├── classroom_behavior_model.h5# Fallback trained model
│   └── class_indices.json         # Class label → index mapping
├── outputs/                       # Generated artifacts
│   ├── anomaly_report.json        # Latest analysis report (JSON)
│   ├── anomaly_report.pdf         # Generated PDF report
│   ├── anomaly_report.csv         # Generated CSV export
│   └── anomaly_snapshots/         # Current run snapshots
│       └── history/               # Archived previous run snapshots
├── logs/
│   ├── classroom_monitor.log      # Runtime / analysis logs
│   └── audit_log.json             # Authentication / system events
├── config/
│   └── config.json                # (Optional) external configuration
├── CNN_Dataset/                   # Training dataset folders per class
├── requirements.txt
├── README.md
└── run_project.bat                # Convenience launcher
```

## 🔧 Key Improvements (Version 2.0)

### ✅ **Bug Fixes**
- **Model Loading**: Fixed memory leaks and inefficient reloading
- **Progress Tracking**: Accurate progress bars showing actual processed frames
- **Memory Management**: Eliminated memory leaks in session state
- **Error Handling**: Comprehensive error handling throughout the application
- **File Naming**: Fixed filename generation for special characters

### 🎨 **UI Enhancements**
- **Modern Design**: Complete UI overhaul with gradient themes and animations
- **Responsive Layout**: Mobile-friendly design with adaptive components
- **Interactive Charts**: Advanced Plotly visualizations with hover effects
- **Professional Styling**: Custom CSS with smooth transitions and effects
- **Analytics Dashboard**: Comprehensive real-time monitoring and reporting

### ⚡ **Performance Optimizations**
- **Configuration System**: External config file for easy parameter tuning
- **Queue Management**: Increased buffer sizes for smooth video processing
- **Memory Optimization**: Reduced memory footprint and improved caching
- **Faster Processing**: Optimized frame processing and model inference

### 🛡️ **Reliability Improvements**
- **Input Validation**: Robust validation for all user inputs
- **Error Recovery**: Graceful error handling with user-friendly messages
- **Logging System**: Comprehensive logging for debugging and monitoring
- **Configuration Management**: Flexible configuration without code changes

## ⚙️ Configuration File (`config.json`)
Minimal example (if using root or `config/config.json`):

```json
{
  "model": {
    "path": "models/classroom_behavior_model.h5",
    "class_indices_path": "models/class_indices.json",
    "image_size": [224, 224],
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50
  },
  "video_analysis": {
    "default_frame_skip": 30,
    "anomaly_threshold": 0.7,
    "max_anomaly_snapshots": 50,
    "snapshot_dir": "outputs/anomaly_snapshots"
  },
  "ui": {
    "theme": "dark",
    "max_displayed_anomalies": 9,
    "chart_height": 400
  }
}
```

If `config.json` is missing, defaults are applied internally.

## 🎮 Usage Guide

### 🚀 **Quick Start**
1. **Activate Environment**:
   ```bash
   cd classroom_monitor_env/Scripts
   activate  # On Windows
   # or
   source activate  # On Linux/Mac
   cd ../..
   ```

2. **Launch Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access Interface**:
   - Open browser to `http://localhost:8501`

### 📚 **Detailed Usage**

#### Training the Model
```bash
# Via Streamlit UI (Recommended)
streamlit run streamlit_app.py
# Navigate to "🤖 Model Training"

# Or via command line
python model_trainer.py
```

#### Video Analysis
```bash
# Via Streamlit UI (Recommended)
streamlit run streamlit_app.py
# Navigate to "🎬 Video Analysis"
# Upload video file and adjust parameters

# Or via command line
python video_analyzer.py
```

#### Dashboard Analytics
```bash
# Via Streamlit UI
streamlit run streamlit_app.py
# Navigate to "📊 Dashboard" for comprehensive analytics
```

## 📊 Model Performance (Illustrative)

- **Accuracy**: 90%+ on test dataset
- **Classes**: 9 behavioral categories
- **Architecture**: Convolutional Neural Network
- **Training Data**: 3,000+ labeled images

## 🧪 Model Loading Logic

Both apps attempt to load the best available model in priority order:
1. `models/best_classroom_model.h5`
2. `models/classroom_behavior_model.h5`
3. Root fallback filenames if above missing.

`class_indices.json` is loaded to build an index→class reverse map. Distribution charts show all classes (zero counts included in bar chart) while the pie chart limits itself to behaviors actually detected.

## 🔧 Adjusting Behavior & Thresholds

The system can be configured for different environments:
- Adjust confidence thresholds in `video_analyzer.py`
- Modify alert sensitivity in `streamlit_app.py`
- Customize behavior categories as needed

## 🔍 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Only one behavior shown | Video content lacks other behaviors | Test with varied sample video |
| Model not found | Files not moved to `models/` | Copy `.h5` + `class_indices.json` into `models/` |
| Empty snapshots folder | No anomalies above threshold | Lower `anomaly_threshold` (e.g. 0.6) |
| Audit log missing | Not yet created | Trigger a login/logout event |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Streamlit for the amazing web framework

## 📧 Contact

 Email - watarenaveen@gmail.com

Project Link: [https://github.com/Naveenexe/anomaly-detection-exam-hall]

---

⭐ **Star this repository if you found it helpful!**

---
### ✅ Next Steps / Roadmap
- Real-time webcam streaming module re-introduction (removed stub `RealTimeAnalyzer`)
- Model confusion matrix & training metrics panel inside enhanced app
- Optional role management UI for password reset
- Lightweight REST API for external integration

---
### 🧪 Quick Windows Commands
```powershell
python -m venv classroom_monitor_env
& "./classroom_monitor_env/Scripts/Activate.ps1"
pip install -r requirements.txt
streamlit run streamlit_app_enhanced.py
```

