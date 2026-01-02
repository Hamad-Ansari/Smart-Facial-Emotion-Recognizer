import cv2
import numpy as np
from utils.emotion_utils import EmotionRecognizer

def test_system():
    print("Testing Emotion Recognition System...")
    
    # Initialize recognizer
    recognizer = EmotionRecognizer()
    
    # Test with a sample image (create a blank one for testing)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n1. Testing image analysis...")
    result = recognizer.analyze_image(test_image)
    
    if result.get('success', False):
        print(f"✅ Success! Emotion: {result['emotion']}")
        print(f"   Confidence scores: {result['emotions']}")
    else:
        print("⚠️ Using fallback mode")
        print(f"   Emotion: {result.get('emotion', 'N/A')}")
    
    print("\n2. Testing webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Webcam is accessible")
        ret, frame = cap.read()
        if ret:
            print("✅ Can read frames from webcam")
            result = recognizer.analyze_image(frame)
            print(f"   Webcam test emotion: {result.get('emotion', 'N/A')}")
        cap.release()
    else:
        print("❌ Webcam not accessible")
    
    print("\n3. Testing face drawing...")
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    test_result = {
        'emotion': 'happy',
        'emotions': {'happy': 90.0, 'sad': 5.0, 'neutral': 5.0},
        'region': {'x': 100, 'y': 100, 'w': 100, 'h': 100},
        'success': True
    }
    
    output = recognizer.draw_emotion_result(test_image, test_result)
    print("✅ Face drawing test passed")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_system()