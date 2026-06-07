"""
Health Data Classifier - Rapid classification using TabPFN

Fast, accurate classification of healthcare tabular data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder
import logging

# Note: TabPFN import (install via: pip install tabpfn)
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TabPFNClassifier = None
    TABPFN_AVAILABLE = False
    logging.warning("TabPFN not available. Using fallback classifier.")

from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)


class HealthDataClassifier:
    """
    Rapid healthcare data classifier using TabPFN or fallback models.
    
    Features:
    - TabPFN for fast, accurate small-data classification
    - Fallback to Random Forest if TabPFN unavailable
    - Handles categorical and numerical features
    - Automatic preprocessing
    """
    
    def __init__(self, use_tabpfn: bool = True):
        """
        Initialize health data classifier.
        
        Args:
            use_tabpfn: Prefer TabPFN if available
        """
        self.use_tabpfn = use_tabpfn and TABPFN_AVAILABLE
        
        if self.use_tabpfn:
            self.model = TabPFNClassifier(device='cpu')
            logger.info("Using TabPFN classifier")
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            logger.warning("Using Random Forest fallback (TabPFN not available)")
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_trained = False
    
    def preprocess_data(
        self,
        data: Union[pd.DataFrame, Dict],
        fit: bool = False
    ) -> np.ndarray:
        """
        Preprocess health data for classification.
        
        Args:
            data: Input data (DataFrame or dict)
            fit: Whether to fit encoders
            
        Returns:
            Numerical feature matrix
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Convert categorical to numerical
        for col in data.select_dtypes(include=['object']).columns:
            if fit:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col].astype(str))
                self.label_encoders[col] = encoder
            elif col in self.label_encoders:
                data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        return data.values
    
    def classify(
        self,
        data: Union[pd.DataFrame, Dict, np.ndarray],
        threshold: float = 0.5
    ) -> Dict:
        """
        Classify health data.
        
        Args:
            data: Input data to classify
            threshold: Confidence threshold
            
        Returns:
            Classification result with confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Preprocess
        if not isinstance(data, np.ndarray):
            X = self.preprocess_data(data)
        else:
            X = data
        
        # Predict
        self.model.eval() if hasattr(self.model, 'eval') else None
        
        if self.use_tabpfn:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        
        result = {
            'prediction': int(predictions[0]),
            'confidence': float(probabilities[0].max()),
            'all_probabilities': probabilities[0].tolist(),
            'threshold_met': probabilities[0].max() > threshold
        }
        
        return result
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ):
        """
        Train the classifier.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Preprocess
        if isinstance(X, pd.DataFrame):
            X_processed = self.preprocess_data(X, fit=True)
        else:
            X_processed = X
        
        # Train
        if self.use_tabpfn and hasattr(self.model, 'fit'):
            self.model.fit(X_processed, y)
        else:
            self.model.fit(X_processed, y)
        
        self.is_trained = True
        logger.info(f"Classifier trained on {len(X)} samples")
    
    def save_model(self, path: str):
        """Save classifier model."""
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'use_tabpfn': self.use_tabpfn,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"Classifier saved to {path}")
    
    def load_model(self, path: str):
        """Load classifier model."""
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoders = data['label_encoders']
        self.use_tabpfn = data['use_tabpfn']
        self.is_trained = data['is_trained']
        logger.info(f"Classifier loaded from {path}")