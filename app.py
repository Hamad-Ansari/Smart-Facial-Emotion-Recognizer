import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from utils.emotion_utils import EmotionRecognizer
import tempfile
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Smart Facial Emotion Recognizer",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize emotion recognizer
@st.cache_resource
def load_emotion_recognizer():
    try:
        return EmotionRecognizer()
    except Exception as e:
        st.error(f"Error loading emotion recognizer: {e}")
        return None

# Custom CSS with better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.4rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .emotion-card:hover {
        transform: translateY(-5px);
    }
    .happy { 
        background: linear-gradient(135deg, #FFF9C4, #FFEB3B);
        color: #F57F17;
        border-left: 6px solid #FFD600;
    }
    .sad { 
        background: linear-gradient(135deg, #E3F2FD, #2196F3);
        color: #0D47A1;
        border-left: 6px solid #1976D2;
    }
    .angry { 
        background: linear-gradient(135deg, #FFEBEE, #F44336);
        color: #B71C1C;
        border-left: 6px solid #D32F2F;
    }
    .surprise { 
        background: linear-gradient(135deg, #F3E5F5, #9C27B0);
        color: #4A148C;
        border-left: 6px solid #7B1FA2;
    }
    .fear { 
        background: linear-gradient(135deg, #E8EAF6, #3F51B5);
        color: #1A237E;
        border-left: 6px solid #303F9F;
    }
    .disgust { 
        background: linear-gradient(135deg, #E8F5E9, #4CAF50);
        color: #1B5E20;
        border-left: 6px solid #388E3C;
    }
    .neutral { 
        background: linear-gradient(135deg, #FAFAFA, #9E9E9E);
        color: #212121;
        border-left: 6px solid #616161;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Smart Facial Emotion Recognizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time facial emotion detection using Computer Vision + AI | 7 Emotion Classes</p>', unsafe_allow_html=True)

# Initialize session state
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'camera' not in st.session_state:
    st.session_state.camera = None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Detection Mode
    st.markdown("#### 📷 Detection Mode")
    detection_mode = st.radio(
        "Choose input method:",
        ["Real-time Webcam", "Upload Image", "Upload Video"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Display Options
    st.markdown("#### 👁️ Display Options")
    show_details = st.checkbox("Show Detailed Analysis", True)
    show_confidence = st.checkbox("Show Confidence Scores", True)
    show_fps = st.checkbox("Show FPS Counter", True)
    
    st.divider()
    
    # Emotion Statistics
    st.markdown("#### 📊 Statistics Control")
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.emotion_history = []
        st.success("Statistics reset successfully!")
    
    if st.button("📥 Export Data", use_container_width=True):
        if st.session_state.emotion_history:
            df = pd.DataFrame(st.session_state.emotion_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="emotion_data.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # Info section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Webcam Mode**: Allow camera access, look at camera
        2. **Image Mode**: Upload clear facial images
        3. **Video Mode**: Upload facial expression videos
        
        **Tips for best results:**
        - Good lighting conditions
        - Face clearly visible
        - Neutral background
        - No sunglasses or masks
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🧠 Tech Stack
    - **Python** 🐍
    - **OpenCV** 👁️
    - **DeepFace** 🎭
    - **Streamlit** 🚀
    """)

# Load emotion recognizer
recognizer = load_emotion_recognizer()

if recognizer is None:
    st.error("""
    ⚠️ **Emotion Recognizer failed to load!**
    
    Please install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    
    If DeepFace fails, the system will use fallback detection.
    """)
    
    # Try to initialize fallback
    recognizer = EmotionRecognizer(backend="fallback")
    st.warning("⚠️ Using fallback emotion detection mode")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Detection area
    if detection_mode == "Real-time Webcam":
        st.markdown("### 🎥 Real-time Emotion Detection")
        
        # Webcam controls
        col_start, col_stop, col_status = st.columns(3)
        
        with col_start:
            if st.button("▶️ Start Webcam", use_container_width=True):
                st.session_state.webcam_active = True
        
        with col_stop:
            if st.button("⏹️ Stop Webcam", use_container_width=True):
                st.session_state.webcam_active = False
                if st.session_state.camera:
                    st.session_state.camera.release()
                    st.session_state.camera = None
        
        with col_status:
            status_color = "🟢" if st.session_state.webcam_active else "🔴"
            st.markdown(f"**Status:** {status_color}")
        
        if st.session_state.webcam_active:
            # Initialize webcam
            if st.session_state.camera is None:
                st.session_state.camera = cv2.VideoCapture(0)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Create placeholders
            frame_placeholder = st.empty()
            fps_placeholder = st.empty()
            emotion_placeholder = st.empty()
            
            fps_counter = []
            
            while st.session_state.webcam_active and st.session_state.camera.isOpened():
                start_time = time.time()
                
                # Read frame
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Analyze emotion
                result = recognizer.analyze_image(frame)
                
                # Draw results
                frame_with_results = recognizer.draw_emotion_result(frame, result)
                
                # Calculate FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter[-30:]) if fps_counter else 0
                
                # Display FPS
                if show_fps:
                    cv2.putText(frame_with_results, f"FPS: {avg_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame_with_results, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Display current emotion
                if result.get('success', False):
                    dominant_emotion = result['emotion']
                    emotion_class = dominant_emotion.lower()
                    
                    with emotion_placeholder.container():
                        st.markdown(f"""
                        <div class="emotion-card {emotion_class}">
                            🎭 Current Emotion: {dominant_emotion.upper()}
                            <br>
                            <small>Confidence: {max(result['emotions'].values()):.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Store in history
                    st.session_state.emotion_history.append({
                        'emotion': dominant_emotion,
                        'timestamp': time.time(),
                        'confidence': max(result['emotions'].values()),
                        'fps': avg_fps
                    })
                
                # Break if stopped
                if not st.session_state.webcam_active:
                    break
            
            # Cleanup
            if not st.session_state.webcam_active and st.session_state.camera:
                st.session_state.camera.release()
                st.session_state.camera = None
                st.success("Webcam stopped successfully!")
        
        else:
            st.info("👆 Click 'Start Webcam' to begin real-time emotion detection!")
            st.image("https://via.placeholder.com/640x480/333/fff?text=Webcam+Preview", 
                    caption="Webcam will appear here", use_column_width=True)
    
    elif detection_mode == "Upload Image":
        st.markdown("### 🖼️ Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a facial image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload a clear image of a face for emotion analysis"
        )
        
        if uploaded_file is not None:
            # Read and display image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert to BGR if needed
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Display original image
            col_orig, col_processed = st.columns(2)
            
            with col_orig:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Analyze emotion
            with st.spinner("🔍 Analyzing facial emotions..."):
                result = recognizer.analyze_image(image_np)
            
            if result.get('success', False):
                # Draw results
                output_image = recognizer.draw_emotion_result(image_np, result)
                output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                
                with col_processed:
                    st.image(output_rgb, caption="Analyzed Image", use_column_width=True)
                
                # Display emotion card
                dominant_emotion = result['emotion']
                emotion_class = dominant_emotion.lower()
                
                st.markdown(f"""
                <div class="emotion-card {emotion_class}">
                    🎯 Emotion Detected: {dominant_emotion.upper()}
                    <br>
                    <small>Confidence: {max(result['emotions'].values()):.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                if show_confidence:
                    st.markdown("#### 📊 Confidence Breakdown")
                    emotions_df = pd.DataFrame(
                        list(result['emotions'].items()),
                        columns=['Emotion', 'Confidence (%)']
                    ).sort_values('Confidence (%)', ascending=False)
                    
                    # Create bar chart
                    fig = px.bar(
                        emotions_df,
                        x='Emotion',
                        y='Confidence (%)',
                        color='Emotion',
                        color_discrete_map={
                            'happy': '#FFD600',
                            'sad': '#2196F3',
                            'angry': '#F44336',
                            'surprise': '#9C27B0',
                            'fear': '#3F51B5',
                            'disgust': '#4CAF50',
                            'neutral': '#9E9E9E'
                        }
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store in history
                st.session_state.emotion_history.append({
                    'emotion': dominant_emotion,
                    'timestamp': time.time(),
                    'confidence': max(result['emotions'].values()),
                    'source': 'uploaded_image'
                })
                
                st.success("✅ Analysis completed successfully!")
            else:
                st.warning("⚠️ No face detected or analysis failed. Please try another image.")
    
    elif detection_mode == "Upload Video":
        st.markdown("### 🎬 Video Analysis")
        
        uploaded_video = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video containing faces for emotion analysis"
        )
        
        if uploaded_video is not None:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Process video
            st.info("Processing video... This may take a moment.")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            emotions_in_video = []
            frame_count = 0
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame for performance
                if frame_count % 10 == 0:
                    result = recognizer.analyze_image(frame)
                    if result.get('success', False):
                        emotions_in_video.append(result['emotion'])
                
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            cap.release()
            os.unlink(video_path)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if emotions_in_video:
                st.success(f"✅ Processed {len(emotions_in_video)} frames with emotion data")
                
                # Emotion distribution
                st.markdown("#### 📈 Emotion Distribution in Video")
                emotion_counts = pd.Series(emotions_in_video).value_counts()
                
                col_chart, col_stats = st.columns([2, 1])
                
                with col_chart:
                    fig = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution",
                        color=emotion_counts.index,
                        color_discrete_map={
                            'happy': '#FFD600',
                            'sad': '#2196F3',
                            'angry': '#F44336',
                            'surprise': '#9C27B0',
                            'fear': '#3F51B5',
                            'disgust': '#4CAF50',
                            'neutral': '#9E9E9E'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_stats:
                    st.markdown("##### 📊 Statistics")
                    st.metric("Total Frames Analyzed", len(emotions_in_video))
                    st.metric("Most Common Emotion", emotion_counts.index[0])
                    st.metric("Unique Emotions", len(emotion_counts))
                
                # Add to history
                for emotion in emotions_in_video[:10]:  # Store first 10
                    st.session_state.emotion_history.append({
                        'emotion': emotion,
                        'timestamp': time.time(),
                        'confidence': 85.0,
                        'source': 'uploaded_video'
                    })
            else:
                st.warning("⚠️ No emotions detected in the video.")

with col2:
    # Analytics panel
    st.markdown("### 📊 Emotion Analytics")
    
    if st.session_state.emotion_history:
        # Create DataFrame
        history_df = pd.DataFrame(st.session_state.emotion_history)
        
        # Convert timestamp to datetime
        if 'timestamp' in history_df.columns:
            history_df['time'] = pd.to_datetime(history_df['timestamp'], unit='s')
        
        # Summary metrics
        st.markdown("##### 📈 Summary")
        col1_metric, col2_metric = st.columns(2)
        
        with col1_metric:
            st.metric("Total Detections", len(history_df))
            if 'emotion' in history_df.columns:
                most_common = history_df['emotion'].mode()[0] if not history_df['emotion'].mode().empty else "N/A"
                st.metric("Most Common", most_common)
        
        with col2_metric:
            if 'confidence' in history_df.columns:
                avg_conf = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            unique_emotions = history_df['emotion'].nunique() if 'emotion' in history_df.columns else 0
            st.metric("Unique Emotions", unique_emotions)
        
        # Emotion timeline
        st.markdown("##### 📅 Emotion Timeline")
        if 'time' in history_df.columns and 'emotion' in history_df.columns:
            timeline_df = history_df.copy()
            timeline_df['minute'] = timeline_df['time'].dt.floor('min')
            
            emotion_timeline = timeline_df.groupby(['minute', 'emotion']).size().unstack(fill_value=0)
            
            fig_timeline = go.Figure()
            for emotion in emotion_timeline.columns:
                fig_timeline.add_trace(go.Scatter(
                    x=emotion_timeline.index,
                    y=emotion_timeline[emotion],
                    mode='lines+markers',
                    name=emotion,
                    stackgroup='one'
                ))
            
            fig_timeline.update_layout(
                title="Emotion Timeline",
                xaxis_title="Time",
                yaxis_title="Count",
                hovermode="x unified"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent detections
        st.markdown("##### 🕐 Recent Detections")
        recent_df = history_df.tail(5).copy()
        if 'timestamp' in recent_df.columns:
            recent_df['time'] = pd.to_datetime(recent_df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
        
        display_cols = ['time', 'emotion', 'confidence'] if 'confidence' in recent_df.columns else ['time', 'emotion']
        display_df = recent_df[display_cols].rename(columns={'time': 'Time', 'emotion': 'Emotion', 'confidence': 'Confidence'})
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export button
        if st.button("📥 Export All Data", use_container_width=True):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="emotion_history.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        # Show placeholder when no data
        st.info("📊 Analytics will appear here")
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 10px;">
            <h3 style="color: #666;">No data yet</h3>
            <p>Start emotion detection to see analytics here!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample visualization
        st.markdown("##### 🎭 Sample Emotion Distribution")
        sample_data = pd.DataFrame({
            'Emotion': ['Happy', 'Sad', 'Neutral', 'Surprise', 'Angry'],
            'Count': [35, 20, 25, 10, 10]
        })
        
        fig_sample = px.bar(
            sample_data,
            x='Emotion',
            y='Count',
            color='Emotion',
            title="Sample Distribution (Demo)"
        )
        st.plotly_chart(fig_sample, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <h4>🧠 Powered by Computer Vision & AI</h4>
    <p>Real-time facial emotion recognition system | 7 Emotion Classes | Live Analytics</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        Made with ❤️ using Python, OpenCV, and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)