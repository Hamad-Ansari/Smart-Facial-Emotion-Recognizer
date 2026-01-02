import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognizer:
    def __init__(self, backend="deepface"):
        self.backend = backend
        self.emotion_colors = {
            'happy': (0, 255, 255),      # Yellow
            'sad': (255, 0, 0),          # Blue
            'angry': (0, 0, 255),        # Red
            'surprise': (255, 255, 0),   # Cyan
            'fear': (255, 0, 255),       # Magenta
            'disgust': (0, 255, 0),      # Green
            'neutral': (255, 255, 255)   # White
        }
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 
                              'sad', 'surprise', 'neutral']
        
        # Try to load DeepFace with fallback
        self.deepface_available = self._check_deepface()
        
        if not self.deepface_available and backend == "deepface":
            print("DeepFace not available. Using Haar Cascade fallback.")
            self._init_haar_cascade()
    
    def _check_deepface(self):
        """Check if DeepFace is available and working"""
        try:
            from deepface import DeepFace
            # Test with a simple call
            test_result = DeepFace.analyze(
                img_path=np.zeros((100, 100, 3), dtype=np.uint8),
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            return True
        except Exception as e:
            print(f"DeepFace initialization error: {e}")
            return False
    
    def _init_haar_cascade(self):
        """Initialize Haar Cascade for face detection"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load emotion model (simplified version)
        self._init_simple_emotion_model()
    
    def _init_simple_emotion_model(self):
        """Initialize a simple emotion model for fallback"""
        # This is a mock emotion model - in production, you'd load a real model
        self.emotion_model_weights = {
            'happy': [0.8, 0.1, 0.1, 0.9, 0.2, 0.1, 0.3],
            'sad': [0.1, 0.1, 0.2, 0.1, 0.9, 0.1, 0.3],
            'angry': [0.9, 0.3, 0.2, 0.1, 0.2, 0.1, 0.2],
            'neutral': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            'surprise': [0.1, 0.1, 0.3, 0.2, 0.1, 0.9, 0.2],
            'fear': [0.2, 0.1, 0.8, 0.1, 0.3, 0.2, 0.1],
            'disgust': [0.3, 0.8, 0.2, 0.1, 0.2, 0.1, 0.1]
        }
    
    def analyze_emotion(self, image_path: str) -> Dict:
        """
        Analyze facial emotion from image path
        """
        try:
            if self.deepface_available:
                from deepface import DeepFace
                
                analysis = DeepFace.analyze(
                    img_path=image_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                return {
                    'emotion': analysis['dominant_emotion'],
                    'emotions': analysis['emotion'],
                    'region': analysis.get('region', {'x': 0, 'y': 0, 'w': 100, 'h': 100}),
                    'success': True
                }
            else:
                # Fallback to simple detection
                return self._analyze_emotion_fallback(image_path)
                
        except Exception as e:
            print(f"Error in analyze_emotion: {e}")
            return self._analyze_emotion_fallback(image_path)
    
    def analyze_image(self, image_array: np.ndarray) -> Dict:
        """
        Analyze emotion from numpy array
        """
        try:
            # Convert to RGB if needed
            if len(image_array.shape) == 2:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            if self.deepface_available:
                from deepface import DeepFace
                
                analysis = DeepFace.analyze(
                    img_path=rgb_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                return {
                    'emotion': analysis['dominant_emotion'],
                    'emotions': analysis['emotion'],
                    'region': analysis.get('region', {'x': 0, 'y': 0, 'w': 100, 'h': 100}),
                    'success': True
                }
            else:
                # Fallback analysis
                return self._analyze_array_fallback(rgb_image)
                
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return self._analyze_array_fallback(image_array)
    
    def _analyze_emotion_fallback(self, image_path: str) -> Dict:
        """Fallback emotion analysis using Haar Cascade"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image")
            
            return self._analyze_array_fallback(img)
            
        except Exception as e:
            print(f"Fallback analysis error: {e}")
            return self._get_default_emotion()
    
    def _analyze_array_fallback(self, image_array: np.ndarray) -> Dict:
        """Fallback analysis for numpy array"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use first face
                x, y, w, h = faces[0]
                
                # Mock emotion prediction (replace with actual model in production)
                # Here we simulate based on facial region characteristics
                face_roi = gray[y:y+h, x:x+w]
                
                # Calculate some simple features
                if face_roi.size > 0:
                    brightness = np.mean(face_roi)
                    contrast = np.std(face_roi)
                    
                    # Simple rule-based emotion (for demo purposes)
                    if brightness > 150:
                        emotion = 'happy'
                    elif brightness < 50:
                        emotion = 'sad'
                    elif contrast > 60:
                        emotion = 'surprise'
                    else:
                        emotion = 'neutral'
                    
                    # Mock confidence scores
                    emotions_dict = {}
                    for label in self.emotion_labels:
                        if label == emotion:
                            emotions_dict[label] = 85.0
                        else:
                            emotions_dict[label] = (100 - 85) / (len(self.emotion_labels) - 1)
                    
                    return {
                        'emotion': emotion,
                        'emotions': emotions_dict,
                        'region': {'x': x, 'y': y, 'w': w, 'h': h},
                        'success': True,
                        'note': 'Using fallback detection'
                    }
            
            # No faces detected
            return {
                'emotion': 'neutral',
                'emotions': {label: 14.28 for label in self.emotion_labels},  # Equal distribution
                'region': {'x': 0, 'y': 0, 'w': 100, 'h': 100},
                'success': False,
                'note': 'No face detected'
            }
            
        except Exception as e:
            print(f"Array fallback error: {e}")
            return self._get_default_emotion()
    
    def _get_default_emotion(self) -> Dict:
        """Return default emotion structure"""
        return {
            'emotion': 'neutral',
            'emotions': {label: 14.28 for label in self.emotion_labels},
            'region': {'x': 0, 'y': 0, 'w': 100, 'h': 100},
            'success': False,
            'note': 'Error in emotion detection'
        }
    
    def draw_emotion_result(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw emotion analysis results on image
        """
        if image is None or image.size == 0:
            return image
        
        # Create a copy of the image
        output = image.copy()
        
        # Get face region
        region = result.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 100)
        h = region.get('h', 100)
        
        # Get dominant emotion
        dominant_emotion = result.get('emotion', 'neutral').lower()
        color = self.emotion_colors.get(dominant_emotion, (255, 255, 255))
        
        # Draw rectangle around face
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        label = f"Emotion: {result.get('emotion', 'Unknown')}"
        cv2.putText(output, label, (x, max(y - 10, 20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw emotion scores if available
        emotions = result.get('emotions', {})
        if emotions:
            y_offset = y + h + 30
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for i, (emotion, score) in enumerate(sorted_emotions):
                text = f"{emotion}: {score:.1f}%"
                cv2.putText(output, text, (x, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add note if present
        if result.get('note'):
            cv2.putText(output, result['note'], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return output
    
    def get_emotion_stats(self, results: List[Dict]) -> pd.DataFrame:
        """
        Get statistics from multiple emotion analyses
        """
        emotions_list = []
        
        for result in results:
            if result.get('success', False):
                emotions = result.get('emotions', {})
                emotions['dominant'] = result.get('emotion', 'unknown')
                emotions_list.append(emotions)
        
        if emotions_list:
            df = pd.DataFrame(emotions_list)
            return df
        return pd.DataFrame()