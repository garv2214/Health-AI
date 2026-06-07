"""
Privacy Utilities - Differential privacy, encryption, and security tools
"""

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import logging

logger = logging.getLogger(__name__)


class PrivacyManager:
    """Utilities for privacy-preserving data processing."""
    
    @staticmethod
    def add_differential_privacy_noise(
        data: np.ndarray,
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            data: Data to privatize
            epsilon: Privacy budget (lower = more privacy)
            sensitivity: Query sensitivity
            
        Returns:
            Noisy data with differential privacy guarantee
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        privatized = data + noise
        
        logger.info(f" Added DP noise (ε={epsilon}, scale={scale:.4f})")
        
        return privatized
    
    @staticmethod
    def encrypt_data(data: bytes, password: str) -> str:
        """Encrypt sensitive data."""
        salt = os.urandom(32)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        # In production: use proper encryption (Fernet/AES)
        encrypted = base64.urlsafe_b64encode(data)
        
        return base64.urlsafe_b64encode(salt + encrypted).decode()
    
    @staticmethod
    def data_anonymization(record: dict) -> dict:
        """
        Anonymize personal health information.
        
        Removes or generalizes direct identifiers.
        """
        anonymized = record.copy()
        
        # Remove direct identifiers
        identifiers = ['name', 'ssn', 'phone', 'email', 'address']
        for field in identifiers:
            if field in anonymized:
                del anonymized[field]
        
        # Generalize age to decade
        if 'age' in anonymized:
            anonymized['age_decade'] = (anonymized['age'] // 10) * 10
            del anonymized['age']
        
        # Generalize location to region
        if 'city' in anonymized:
            anonymized['region'] = 'REDACTED'
            del anonymized['city']
        
        return anonymized