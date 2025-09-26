# 🎯 Anomaly Detection in Exam Hall

An AI-powered real-time academic integrity monitoring system that uses computer vision and deep learning to detect suspicious behaviors during examinations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)

## 🚀 Features

- **Real-time Behavior Detection**: Monitors 9 different types of academic misconduct behaviors
- **CNN-based Classification**: Deep learning model with high accuracy detection
- **Live Video Analysis**: Real-time processing using webcam or uploaded videos
- **Professional Web Interface**: Multi-page Streamlit application with interactive dashboards
- **Automated Alert System**: Instant notifications to examination supervisors
- **Comprehensive Reporting**: Detailed analytics with confidence scoring and timestamps
- **Anomaly Snapshots**: Automatic capture of suspicious activities

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

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras, CNN
- **Computer Vision**: OpenCV, Image Processing
- **Web Framework**: Streamlit
- **Data Visualization**: Plotly
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for real-time monitoring)
- GPU recommended for faster processing

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/anomaly-detection-exam-hall.git
   cd anomaly-detection-exam-hall
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── model_trainer.py          # CNN model training script
├── video_analyzer.py         # Video analysis and processing
├── requirements.txt          # Python dependencies
├── best_classroom_model.h5   # Trained CNN model
├── class_indices.json        # Class label mappings
├── confusion_matrix.png      # Model performance visualization
└── README.md                # Project documentation
```

## 🎮 Usage

### Training the Model
```bash
python model_trainer.py
```

### Video Analysis
```bash
python video_analyzer.py --input your_video.mp4
```

### Web Application
```bash
streamlit run streamlit_app.py
```

## 📊 Model Performance

- **Accuracy**: 90%+ on test dataset
- **Classes**: 9 behavioral categories
- **Architecture**: Convolutional Neural Network
- **Training Data**: 3,000+ labeled images

## 🔧 Configuration

The system can be configured for different environments:
- Adjust confidence thresholds in `video_analyzer.py`
- Modify alert sensitivity in `streamlit_app.py`
- Customize behavior categories as needed

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

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/anomaly-detection-exam-hall](https://github.com/yourusername/anomaly-detection-exam-hall)

---

⭐ **Star this repository if you found it helpful!**
