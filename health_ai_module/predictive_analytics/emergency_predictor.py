"""
Emergency Predictor - Uses pre-trained models for emergency prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

logger = logging.getLogger(__name__)


class EmergencyPredictor:
    """Production-ready emergency predictor with pre-trained models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize with pre-trained model."""
        self.model_path = model_path or 'models/emergency_model.pkl'
        
        # Load pre-trained model
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
            logger.info(f"✅ Loaded pre-trained model from {self.model_path}")
        else:
            logger.warning("⚠️  No pre-trained model found. Train first using train_model.py")
            self.is_trained = False
    
    def preprocess_data(self, health_data: Dict) -> np.ndarray:
        """Preprocess health data."""
        features = self._extract_features(health_data)
        features_array = np.array(features).reshape(1, -1)
        return self.scaler.transform(features_array)
    
    def _extract_features(self, health_data: Dict) -> List[float]:
        """Extract features from health data."""
        bp = health_data.get('blood_pressure', '120/80')
        systolic = int(bp.split('/')[0]) if isinstance(bp, str) else bp
        diastolic = int(bp.split('/')[1]) if isinstance(bp, str) else bp
        
        return [
            health_data.get('age', 40),
            health_data.get('weight_kg', 70),
            health_data.get('heart_rate', 80),
            systolic,
            diastolic,
            health_data.get('oxygen_saturation', 98),
            health_data.get('temperature', 37.0),
            health_data.get('respiration_rate', 16),
            health_data.get('pain_level', 0),
            health_data.get('consciousness_score', 10),
            health_data.get('num_conditions', 0)
        ]
    
    def predict_emergency(self, health_data: Dict) -> Tuple[float, str, Dict]:
        """Predict emergency risk."""
        if not self.is_trained:
            raise ValueError("Model not trained. Run train_model.py first.")
        
        scaled_features = self.preprocess_data(health_data)
        
        # Use best model
        best_model = self.models[self.best_model_name]
        
        risk_score = best_model.predict_proba(scaled_features)[0][1]
        prediction = 'EMERGENCY' if risk_score > 0.5 else 'STABLE'
        
        risk_level = 'CRITICAL' if risk_score > 0.8 else 'HIGH' if risk_score > 0.6 else 'MODERATE' if risk_score > 0.4 else 'LOW'
        
        explanation = {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'prediction': prediction,
            'confidence': float(best_model.predict_proba(scaled_features)[0].max())
        }
        
        return risk_score, prediction, explanation
    
    def load_model(self, path: str):
        """Load pre-trained model."""
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.results = data['results']
        self.best_model_name = data['best_model']
        self.is_trained = True
    
    def save_model(self, path: str):
        """Save model."""
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'results': self.results,
            'best_model': self.best_model_name
        }, path)