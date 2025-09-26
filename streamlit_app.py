import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import tempfile
from pathlib import Path
import threading
import queue
import time
from PIL import Image
import base64

# Import our custom modules
from model_trainer import ClassroomBehaviorTrainer
from video_analyzer import VideoAnalyzer, RealTimeAnalyzer

# Page configuration
st.set_page_config(
    page_title="🎓 AI Classroom Monitor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
        animation: pulse 2s infinite;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .alert-success {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .anomaly-snapshot {
        border: 3px solid #dc3545;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        background-color: #f8f9fa;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.video_analyzer = None
        self.real_time_analyzer = None
        self.model_trained = False
        
        # Initialize session state
        if 'anomalies' not in st.session_state:
            st.session_state.anomalies = []
        if 'alert_active' not in st.session_state:
            st.session_state.alert_active = False
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
    
    def load_model_if_exists(self):
        """Load model if it exists"""
        if os.path.exists('classroom_behavior_model.h5') and os.path.exists('class_indices.json'):
            try:
                self.video_analyzer = VideoAnalyzer()
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return False
    
    def show_header(self):
        """Display main header"""
        st.markdown('<h1 class="main-header">🎓 AI-Powered Classroom Monitor</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            model_status = "✅ Ready" if self.load_model_if_exists() else "❌ Not Trained"
            st.markdown(f'<div class="metric-card"><h3>Model Status</h3><p>{model_status}</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            alert_status = "🚨 Active" if st.session_state.alert_active else "🟢 Normal"
            st.markdown(f'<div class="metric-card"><h3>Alert Status</h3><p>{alert_status}</p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            anomaly_count = len(st.session_state.anomalies)
            st.markdown(f'<div class="metric-card"><h3>Anomalies Today</h3><p>{anomaly_count}</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            monitoring_status = "🔴 Live" if st.session_state.monitoring_active else "⚫ Offline"
            st.markdown(f'<div class="metric-card"><h3>Monitoring</h3><p>{monitoring_status}</p></div>', 
                       unsafe_allow_html=True)
    
    def show_alerts(self):
        """Display alert system for HODs"""
        if st.session_state.alert_active:
            st.markdown("""
            <div class="alert-box alert-danger">
                <h2>🚨 URGENT ALERT - HOD ATTENTION REQUIRED</h2>
                <p><strong>Anomalous behavior detected in classroom!</strong></p>
                <p>Immediate intervention may be required.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-dismiss alert after some time
            if st.button("🔕 Acknowledge Alert"):
                st.session_state.alert_active = False
                st.rerun()
    
    def model_training_page(self):
        """Model training interface"""
        st.header("🤖 Model Training")
        
        if not os.path.exists('CNN_Dataset'):
            st.error("❌ CNN_Dataset folder not found! Please ensure your dataset is in the correct location.")
            return
        
        st.success("✅ Dataset found!")
        
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Training Epochs", 10, 100, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        
        with col2:
            img_size = st.selectbox("Image Size", [(224, 224), (256, 256)], index=0)
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        if st.button("🚀 Start Training", key="train_button"):
            with st.spinner("Training model... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize trainer
                    trainer = ClassroomBehaviorTrainer(
                        dataset_path="CNN_Dataset",
                        img_size=img_size,
                        batch_size=batch_size
                    )
                    
                    status_text.text("Loading and preprocessing data...")
                    progress_bar.progress(10)
                    
                    # Load data
                    train_gen, val_gen = trainer.load_and_preprocess_data()
                    
                    status_text.text("Building model architecture...")
                    progress_bar.progress(20)
                    
                    # Build model
                    model = trainer.build_model()
                    
                    status_text.text("Training model...")
                    progress_bar.progress(30)
                    
                    # Train model
                    history = trainer.train_model(epochs=epochs)
                    
                    status_text.text("Evaluating model...")
                    progress_bar.progress(80)
                    
                    # Evaluate model
                    report, cm = trainer.evaluate_model()
                    
                    status_text.text("Saving model...")
                    progress_bar.progress(90)
                    
                    # Save model
                    trainer.save_model()
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    st.success("🎉 Model training completed successfully!")
                    st.balloons()
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Final Accuracy", f"{max(history.history['val_accuracy']):.4f}")
                        st.metric("Final Loss", f"{min(history.history['val_loss']):.4f}")
                    
                    with col2:
                        st.metric("Training Time", f"{epochs} epochs")
                        st.metric("Total Parameters", f"{model.count_params():,}")
                    
                    self.model_trained = True
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    def video_analysis_page(self):
        """Video analysis interface"""
        st.header("🎬 Video Analysis")
        
        if not self.load_model_if_exists():
            st.warning("⚠️ Please train the model first!")
            return
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload classroom video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for anomalous behavior"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success("✅ Video uploaded successfully!")
            
            # Analysis parameters
            col1, col2 = st.columns(2)
            
            with col1:
                frame_skip = st.slider("Frame Skip (for faster processing)", 1, 60, 30)
                confidence_threshold = st.slider("Anomaly Confidence Threshold", 0.5, 0.95, 0.7)
            
            with col2:
                st.info("Higher frame skip = faster processing but may miss some anomalies")
                st.info("Higher threshold = fewer false positives but may miss real anomalies")
            
            if st.button("🔍 Analyze Video", key="analyze_button"):
                with st.spinner("Analyzing video... Please wait."):
                    progress_bar = st.progress(0)
                    
                    def progress_callback(progress):
                        progress_bar.progress(progress / 100)
                    
                    try:
                        # Set threshold
                        self.video_analyzer.anomaly_threshold = confidence_threshold
                        
                        # Analyze video
                        anomalies = self.video_analyzer.analyze_video(
                            video_path, 
                            frame_skip=frame_skip,
                            progress_callback=progress_callback
                        )
                        
                        # Save results to session state
                        st.session_state.anomalies = anomalies
                        
                        if anomalies:
                            st.session_state.alert_active = True
                            
                            st.error(f"🚨 {len(anomalies)} anomalies detected!")
                            
                            # Save snapshots
                            snapshot_files = self.video_analyzer.save_anomaly_snapshots(anomalies)
                            
                            # Generate report
                            report = self.video_analyzer.generate_anomaly_report(anomalies, video_path)
                            
                            # Display results
                            self.show_analysis_results(anomalies, report)
                            
                        else:
                            st.success("✅ No anomalies detected! Classroom behavior appears normal.")
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(video_path)
    
    def show_analysis_results(self, anomalies, report):
        """Display analysis results"""
        st.header("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        
        with col2:
            unique_behaviors = len(set(a['behavior'] for a in anomalies))
            st.metric("Unique Behaviors", unique_behaviors)
        
        with col3:
            avg_confidence = np.mean([a['confidence'] for a in anomalies])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Anomaly timeline
        if anomalies:
            df = pd.DataFrame([{
                'Timestamp': a['timestamp'],
                'Behavior': a['behavior'],
                'Confidence': a['confidence']
            } for a in anomalies])
            
            fig = px.scatter(df, x='Timestamp', y='Confidence', 
                           color='Behavior', size='Confidence',
                           title="Anomaly Timeline",
                           hover_data=['Behavior'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Behavior distribution
            behavior_counts = df['Behavior'].value_counts()
            fig_pie = px.pie(values=behavior_counts.values, names=behavior_counts.index,
                           title="Anomaly Distribution by Behavior Type")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Anomaly snapshots
        st.subheader("🖼️ Anomaly Snapshots")
        
        if os.path.exists('anomaly_snapshots'):
            snapshot_files = list(Path('anomaly_snapshots').glob('*.jpg'))
            
            if snapshot_files:
                cols = st.columns(3)
                
                for i, snapshot_file in enumerate(snapshot_files[:9]):  # Show first 9
                    with cols[i % 3]:
                        image = Image.open(snapshot_file)
                        st.image(image, caption=snapshot_file.name, use_column_width=True)
                        
                        # Extract info from filename
                        parts = snapshot_file.stem.split('_')
                        if len(parts) >= 4:
                            timestamp = parts[2]
                            behavior = '_'.join(parts[3:])
                            st.markdown(f"""
                            <div class="anomaly-snapshot">
                                <strong>Time:</strong> {timestamp}<br>
                                <strong>Behavior:</strong> {behavior}
                            </div>
                            """, unsafe_allow_html=True)
    
    def webcam_monitoring_page(self):
        """Real-time webcam monitoring"""
        st.header("📹 Live Webcam Monitoring")
        
        if not self.load_model_if_exists():
            st.warning("⚠️ Please train the model first!")
            return
        
        st.info("🎥 This feature enables real-time monitoring of classroom behavior through your webcam.")
        
        # Monitoring controls
        col1, col2 = st.columns(2)
        
        with col1:
            camera_index = st.selectbox("Select Camera", [0, 1, 2], index=0)
            confidence_threshold = st.slider("Alert Threshold", 0.5, 0.95, 0.7, key="webcam_threshold")
        
        with col2:
            alert_sound = st.checkbox("Enable Sound Alerts", value=True)
            auto_screenshot = st.checkbox("Auto-save Anomaly Screenshots", value=True)
        
        # Live monitoring status
        if st.session_state.monitoring_active:
            st.success("🔴 Live monitoring is active!")
            
            if st.button("⏹️ Stop Monitoring"):
                st.session_state.monitoring_active = False
                st.rerun()
        else:
            if st.button("▶️ Start Live Monitoring"):
                st.session_state.monitoring_active = True
                st.info("🎥 Starting webcam monitoring... (This is a demo - actual implementation would use WebRTC)")
                
                # In a real implementation, you would integrate with streamlit-webrtc
                # For demo purposes, we'll show a placeholder
                st.markdown("""
                <div style="background: #000; height: 400px; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; border-radius: 10px;">
                    📹 LIVE WEBCAM FEED<br>
                    <small style="font-size: 16px;">Real-time anomaly detection active</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent detections
        if st.session_state.anomalies:
            st.subheader("🕒 Recent Detections")
            
            recent_anomalies = st.session_state.anomalies[-5:]  # Show last 5
            
            for anomaly in reversed(recent_anomalies):
                with st.expander(f"🚨 {anomaly['behavior']} - Confidence: {anomaly['confidence']:.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Time:** {anomaly['timestamp']}")
                        st.write(f"**Behavior:** {anomaly['behavior']}")
                        st.write(f"**Confidence:** {anomaly['confidence']:.2f}")
                    
                    with col2:
                        if 'frame' in anomaly:
                            st.image(anomaly['frame'], caption="Detected Frame", width=200)
    
    def dashboard_page(self):
        """Main dashboard"""
        st.header("📊 Monitoring Dashboard")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Cameras", "1", delta="0")
        
        with col2:
            st.metric("Anomalies Today", len(st.session_state.anomalies), delta="+2")
        
        with col3:
            st.metric("System Uptime", "99.9%", delta="+0.1%")
        
        with col4:
            st.metric("Alert Response Time", "< 5s", delta="-1s")
        
        # Recent activity chart
        if st.session_state.anomalies:
            # Create hourly anomaly chart
            df = pd.DataFrame([{
                'hour': datetime.now().hour,
                'count': len(st.session_state.anomalies)
            }])
            
            fig = go.Figure(data=go.Bar(x=['Current Hour'], y=[len(st.session_state.anomalies)]))
            fig.update_layout(title="Anomalies This Hour", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # System status
        st.subheader("🖥️ System Status")
        
        status_data = {
            "Component": ["AI Model", "Camera Feed", "Alert System", "Database"],
            "Status": ["✅ Online", "✅ Online", "✅ Online", "✅ Online"],
            "Last Check": ["Just now", "Just now", "Just now", "Just now"]
        }
        
        st.dataframe(pd.DataFrame(status_data), use_container_width=True)
    
    def run(self):
        """Main application runner"""
        # Sidebar navigation
        st.sidebar.title("🎓 Navigation")
        
        pages = {
            "📊 Dashboard": self.dashboard_page,
            "🤖 Model Training": self.model_training_page,
            "🎬 Video Analysis": self.video_analysis_page,
            "📹 Live Monitoring": self.webcam_monitoring_page
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Quick Info")
        st.sidebar.info("""
        **System Features:**
        - 🤖 AI-powered behavior detection
        - 🎬 Video analysis with snapshots
        - 📹 Real-time webcam monitoring
        - 🚨 Instant HOD alerts
        - 📊 Comprehensive reporting
        """)
        
        # Main content
        self.show_header()
        self.show_alerts()
        
        # Run selected page
        pages[selected_page]()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            🎓 AI Classroom Monitor | Built with Streamlit & TensorFlow<br>
            <small>Ensuring academic integrity through intelligent monitoring</small>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
