from transformers import pipeline
from typing import Dict, List, Tuple, Optional, Any
import torch
from pathlib import Path
import json
from datetime import datetime
from advanced_crisis_detector import AdvancedCrisisDetector
from typing import Optional

class EmotionDetector:
    """
    A class to detect emotions in text using a pre-trained model from Hugging Face.
    This version uses a mental health-focused emotion model.
    """
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        """
        Initialize the emotion detector with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model from Hugging Face.
                       Default is 'SamLowe/roberta-base-go_emotions' which is trained on
                       mental health-related text and provides more nuanced emotional analysis.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading emotion detection model: {model_name}")
        print(f"Device set to use {self.device}")
        
        try:
            # Load model without model_kwargs to avoid revision conflict
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=5,  # Limit to top 5 emotions for better performance
                device=self.device
            )
            print(f"Emotion detection model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading emotion detection model: {str(e)}")
            print("Falling back to default emotion model...")
            # Use a simpler model as fallback
            self.classifier = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                top_k=5,
                device=self.device
            )
        
        self.emotion_history = {}
        # Map model-specific labels to standard emotion categories
        self.emotion_mapping = {
            'admiration': 'admiration',
            'amusement': 'joy',
            'anger': 'anger',
            'annoyance': 'annoyance',
            'approval': 'approval',
            'caring': 'caring',
            'confusion': 'confusion',
            'curiosity': 'curiosity',
            'desire': 'desire',
            'disappointment': 'sadness',
            'disapproval': 'disapproval',
            'disgust': 'disgust',
            'embarrassment': 'embarrassment',
            'excitement': 'excitement',
            'fear': 'fear',
            'gratitude': 'gratitude',
            'grief': 'sadness',
            'joy': 'joy',
            'love': 'love',
            'nervousness': 'anxiety',
            'optimism': 'optimism',
            'pride': 'pride',
            'realization': 'realization',
            'relief': 'relief',
            'remorse': 'remorse',
            'sadness': 'sadness',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }
        
    def detect_emotion(self, text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect emotions in the given text with a focus on mental health context.
        
        Args:
            text: Input text to analyze
            user_id: Optional user ID to track emotion history
            
        Returns:
            Dictionary with emotion scores mapped to standard emotion categories
        """
        if not text.strip():
            return {}
            
        try:
            # Get emotion predictions
            predictions = self.classifier(text)[0]
            
            # Process predictions and map to standard emotion categories
            emotion_scores = {}
            for pred in predictions:
                label = pred['label'].lower()
                score = float(pred['score'])
                
                # Map to standard emotion category if mapping exists
                category = self.emotion_mapping.get(label, label)
                
                # Sum scores for the same category from different labels
                if category in emotion_scores:
                    emotion_scores[category] = max(emotion_scores[category], score)
                else:
                    emotion_scores[category] = score
            
            # Normalize scores to sum to 1
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            
            # Update emotion history if user_id is provided
            if user_id is not None:
                # Ensure user_id is a string
                user_id_str = str(user_id)
                
                # Initialize user's emotion history if it doesn't exist
                if user_id_str not in self.emotion_history:
                    self.emotion_history[user_id_str] = []
                
                # Add new emotion data
                if emotion_scores:  # Only add if we have valid emotion scores
                    self.emotion_history[user_id_str].append({
                        'timestamp': datetime.now().isoformat(),
                        'text': text,
                        'emotions': emotion_scores,
                        'dominant_emotion': max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
                    })
                    
                    # Keep only the last 100 entries per user to prevent memory issues
                    if len(self.emotion_history[user_id_str]) > 100:
                        self.emotion_history[user_id_str] = self.emotion_history[user_id_str][-100:]
                    
            # Initialize crisis detector and check for crisis
            crisis_detector = AdvancedCrisisDetector()
            crisis_info = crisis_detector.detect_crisis(text)
            
            # If crisis detected, adjust emotion scores
            if crisis_info['is_crisis'] or crisis_info['is_warning']:
                # Boost negative emotions in crisis situations
                for emotion in ['sadness', 'fear', 'anxiety']:
                    if emotion in emotion_scores:
                        emotion_scores[emotion] = min(1.0, emotion_scores[emotion] * 1.5)
            
            result = {
                'emotions': emotion_scores if emotion_scores else {'neutral': 1.0},
                'crisis_info': crisis_info,
                'dominant_emotion': max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
            }
            
            return result
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            # Return neutral as default if detection fails
            return {'neutral': 1.0}
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """
        Get the dominant emotion and its score for the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (dominant_emotion, score)
        """
        emotion_scores = self.detect_emotion(text)
        if not emotion_scores:
            return "neutral", 1.0
        return max(emotion_scores.items(), key=lambda x: x[1])
    
    def get_emotion_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        Get emotion history for a specific user.
        
        Args:
            user_id: User ID to get history for
            limit: Maximum number of history entries to return
            
        Returns:
            List of emotion history entries
        """
        if not user_id:
            return []
            
        # Ensure user_id is a string for consistent dictionary key access
        user_id_str = str(user_id)
        
        if user_id_str not in self.emotion_history:
            return []
            
        # Return the most recent entries, up to the limit
        return self.emotion_history[user_id_str][-limit:]
    
    def save_emotion_history(self, filepath: str):
        """
        Save emotion history to a JSON file.
        
        Args:
            filepath: Path to save the emotion history
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.emotion_history, f, indent=2)
        except Exception as e:
            print(f"Error saving emotion history: {str(e)}")
    
    def load_emotion_history(self, filepath: str):
        """
        Load emotion history from a JSON file.
        
        Args:
            filepath: Path to load the emotion history from
        """
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    self.emotion_history = json.load(f)
        except Exception as e:
            print(f"Error loading emotion history: {str(e)}")

# Create a global instance for easy import
emotion_detector = EmotionDetector()
