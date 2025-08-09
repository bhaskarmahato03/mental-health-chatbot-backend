"""Advanced Crisis Detection System
Combines rule-based patterns with ML-based sentiment analysis."""

import re
from typing import Dict, List, Any
from transformers import pipeline

class AdvancedCrisisDetector:
    def __init__(self):
        # Use a lightweight sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU for stability
        )
        self.crisis_phrases = [r'kill\s*(my)?self', r'end\s*(it\s*all|my\s*life)', 'suicid']
        self.warning_phrases = ['hopeless', 'helpless', 'worthless', 'no one cares']
        self.safe_phrases = ['struggling with', 'hard time', 'stress about']
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            return {'sentiment': result['label'].lower(), 'score': float(result['score'])}
        except:
            return {'sentiment': 'neutral', 'score': 0.5}
            
    def detect_crisis(self, text: str) -> Dict[str, Any]:
        if not text:
            return self._default_response()
            
        text_lower = text.lower()
        
        # Check safe phrases first
        if any(phrase in text_lower for phrase in self.safe_phrases):
            return self._default_response()
            
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Check for crisis/warning phrases
        crisis_match = any(re.search(p, text_lower) for p in self.crisis_phrases)
        warning_match = any(phrase in text_lower for phrase in self.warning_phrases)
        
        # Calculate risk score (0-1)
        risk_score = 0.0
        if sentiment['sentiment'] == 'negative':
            risk_score += sentiment['score'] * 0.4
        if crisis_match:
            risk_score += 0.5
        if warning_match:
            risk_score += 0.2
            
        risk_score = min(1.0, risk_score)
        
        if risk_score >= 0.7 or crisis_match:
            return {
                'is_crisis': True,
                'is_warning': False,  # Explicitly set is_warning
                'risk_level': 'high',
                'message': "I'm concerned about what you're sharing. Please contact Vandrevala (1860-2662-345) or iCall (9152987821)."
            }
        elif risk_score >= 0.4 or warning_match:
            return {
                'is_crisis': False,
                'is_warning': True,  # Set is_warning for medium risk
                'risk_level': 'medium',
                'message': "I hear you're going through a tough time. Would you like to talk about it?"
            }
        return self._default_response()
    
    def _default_response(self) -> Dict[str, Any]:
        return {
            'is_crisis': False,
            'is_warning': False,  # Add is_warning to default response
            'risk_level': 'low',
            'message': ''
        }
