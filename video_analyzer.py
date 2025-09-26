import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import threading
import queue

class VideoAnalyzer:
    def __init__(self, model_path='classroom_behavior_model.h5', class_indices_path='class_indices.json'):
        self.model = None
        self.class_indices = {}
        self.classes = []
        self.img_size = (224, 224)
        self.anomaly_threshold = 0.7  # Confidence threshold for anomaly detection
        self.normal_class = 'Normal'
        
        # Load model and classes
        self.load_model(model_path, class_indices_path)
        
    def load_model(self, model_path, class_indices_path):
        """Load the trained model and class indices"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
            
            # Create reverse mapping (index to class name)
            self.classes = list(self.class_indices.keys())
            self.idx_to_class = {v: k for k, v in self.class_indices.items()}
            
            print(f"✅ Classes loaded: {self.classes}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize frame
        frame_resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_behavior(self, frame):
        """Predict behavior from a single frame"""
        if self.model is None:
            return None, 0.0
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = self.idx_to_class[predicted_idx]
        
        return predicted_class, confidence
    
    def is_anomaly(self, predicted_class, confidence):
        """Determine if the prediction indicates an anomaly"""
        if predicted_class == self.normal_class:
            return False
        
        return confidence >= self.anomaly_threshold
    
    def analyze_video(self, video_path, frame_skip=30, progress_callback=None):
        """Analyze video and detect anomalies"""
        print(f"🎬 Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
        
        anomalies = []
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_count % frame_skip == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                timestamp_str = str(timedelta(seconds=int(timestamp)))
                
                # Predict behavior
                predicted_class, confidence = self.predict_behavior(frame)
                
                # Check for anomaly
                if self.is_anomaly(predicted_class, confidence):
                    anomaly_data = {
                        'frame_number': frame_count,
                        'timestamp': timestamp_str,
                        'timestamp_seconds': timestamp,
                        'behavior': predicted_class,
                        'confidence': confidence,
                        'frame': frame.copy()
                    }
                    anomalies.append(anomaly_data)
                    print(f"🚨 Anomaly detected at {timestamp_str}: {predicted_class} ({confidence:.2f})")
                
                processed_frames += 1
                
                # Progress callback
                if progress_callback:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Analysis complete! Found {len(anomalies)} anomalies in {processed_frames} processed frames")
        return anomalies
    
    def save_anomaly_snapshots(self, anomalies, output_dir='anomaly_snapshots'):
        """Save anomaly frames as images"""
        if not anomalies:
            print("No anomalies to save")
            return []
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        saved_files = []
        
        for i, anomaly in enumerate(anomalies):
            # Create filename
            timestamp_clean = anomaly['timestamp'].replace(':', '-')
            filename = f"anomaly_{i+1:03d}_{timestamp_clean}_{anomaly['behavior']}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, anomaly['frame'])
            saved_files.append(filepath)
            
            print(f"💾 Saved: {filename}")
        
        print(f"✅ Saved {len(saved_files)} anomaly snapshots to {output_dir}")
        return saved_files
    
    def generate_anomaly_report(self, anomalies, video_path, output_file='anomaly_report.json'):
        """Generate detailed anomaly report"""
        if not anomalies:
            print("No anomalies to report")
            return
        
        # Prepare report data
        report = {
            'video_path': video_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_anomalies': len(anomalies),
            'anomaly_summary': {},
            'detailed_anomalies': []
        }
        
        # Count anomalies by type
        for anomaly in anomalies:
            behavior = anomaly['behavior']
            if behavior not in report['anomaly_summary']:
                report['anomaly_summary'][behavior] = 0
            report['anomaly_summary'][behavior] += 1
            
            # Add to detailed list (without frame data)
            detailed_anomaly = {
                'frame_number': anomaly['frame_number'],
                'timestamp': anomaly['timestamp'],
                'timestamp_seconds': anomaly['timestamp_seconds'],
                'behavior': anomaly['behavior'],
                'confidence': anomaly['confidence']
            }
            report['detailed_anomalies'].append(detailed_anomaly)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Anomaly report saved to {output_file}")
        
        # Print summary
        print("\n📊 ANOMALY SUMMARY:")
        print("-" * 30)
        for behavior, count in report['anomaly_summary'].items():
            print(f"{behavior}: {count} occurrences")
        
        return report

class RealTimeAnalyzer:
    def __init__(self, model_path='classroom_behavior_model.h5', class_indices_path='class_indices.json'):
        self.video_analyzer = VideoAnalyzer(model_path, class_indices_path)
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
    def start_webcam_analysis(self, camera_index=0):
        """Start real-time webcam analysis"""
        print("🎥 Starting real-time webcam analysis...")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Error opening camera {camera_index}")
            return
        
        self.is_running = True
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self._analysis_worker)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Display frame with predictions
                display_frame = self._draw_predictions(frame)
                cv2.imshow('Classroom Monitoring', display_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Stopping analysis...")
        
        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
    
    def _analysis_worker(self):
        """Worker thread for frame analysis"""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Analyze frame
                predicted_class, confidence = self.video_analyzer.predict_behavior(frame)
                
                # Put result in queue
                result = {
                    'class': predicted_class,
                    'confidence': confidence,
                    'is_anomaly': self.video_analyzer.is_anomaly(predicted_class, confidence),
                    'timestamp': datetime.now()
                }
                
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def _draw_predictions(self, frame):
        """Draw predictions on frame"""
        display_frame = frame.copy()
        
        try:
            # Get latest result
            result = self.result_queue.get_nowait()
            
            # Prepare text
            text = f"{result['class']}: {result['confidence']:.2f}"
            color = (0, 0, 255) if result['is_anomaly'] else (0, 255, 0)  # Red for anomaly, Green for normal
            
            # Draw background rectangle
            cv2.rectangle(display_frame, (10, 10), (400, 60), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw anomaly indicator
            if result['is_anomaly']:
                cv2.putText(display_frame, "ANOMALY DETECTED!", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        except queue.Empty:
            # No new results
            pass
        
        return display_frame

def main():
    """Test the video analyzer"""
    analyzer = VideoAnalyzer()
    
    # Test with a video file (replace with your video path)
    video_path = "test_video.mp4"
    
    if os.path.exists(video_path):
        # Analyze video
        anomalies = analyzer.analyze_video(video_path)
        
        # Save snapshots
        analyzer.save_anomaly_snapshots(anomalies)
        
        # Generate report
        analyzer.generate_anomaly_report(anomalies, video_path)
    else:
        print("No test video found. Starting real-time analysis...")
        
        # Start real-time analysis
        real_time = RealTimeAnalyzer()
        real_time.start_webcam_analysis()

if __name__ == "__main__":
    main()
