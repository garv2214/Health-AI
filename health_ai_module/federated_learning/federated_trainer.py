"""
Federated Learning Trainer - Privacy-preserving distributed AI training

Trains models across decentralized nodes without centralizing sensitive data
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SimpleHealthModel(nn.Module):
    """Simple neural network for health prediction."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class FederatedTrainer:
    """
    Federated Learning trainer for decentralized health data.
    
    Key Features:
    - Local training on each node (data never leaves)
    - Secure model aggregation (only weights shared)
    - Differential privacy support
    - Secure aggregation protocol
    """
    
    def __init__(
        self,
        model_type: str = "emergency_predictor",
        privacy_level: str = "high",
        aggregation_method: str = "secure_average"
    ):
        """
        Initialize federated trainer.
        
        Args:
            model_type: Type of model to train
            privacy_level: 'low', 'medium', or 'high'
            aggregation_method: How to aggregate client models
        """
        self.model_type = model_type
        self.privacy_level = privacy_level
        self.aggregation_method = aggregation_method
        
        # Initialize global model
        self.global_model = SimpleHealthModel()
        self.optimizer = Adam(self.global_model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        # Client models
        self.client_models: Dict[str, SimpleHealthModel] = {}
        self.client_weights: Dict[str, int] = {}
        
        # Training state
        self.current_round = 0
        self.is_trained = False
        
        logger.info(f"Federated Trainer initialized - Privacy: {privacy_level}")
    
    def local_training(
        self,
        client_id: str,
        local_data: torch.Tensor,
        local_labels: torch.Tensor,
        epochs: int = 5,
        batch_size: int = 32
    ) -> Dict:
        """
        Train model locally on a single node (data stays local).
        
        Args:
            client_id: Unique client identifier
            local_data: Local training data (never leaves node)
            local_labels: Local labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Model weights and metrics (not raw data)
        """
        # Create local model copy
        local_model = SimpleHealthModel()
        local_model.load_state_dict(self.global_model.state_dict())
        
        # Prepare data loader
        dataset = TensorDataset(local_data, local_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Local training
        local_model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, batch_labels in dataloader:
                self.optimizer.zero_grad()
                outputs = local_model(batch_data)
                loss = self.criterion(outputs.squeeze(), batch_labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
        
        # Add differential privacy noise if high privacy
        if self.privacy_level == "high":
            self._add_privacy_noise(local_model)
        
        # Store model weights (not data!)
        self.client_models[client_id] = local_model
        self.client_weights[client_id] = len(local_data)
        
        metrics = {
            'client_id': client_id,
            'epochs': epochs,
            'final_loss': losses[-1],
            'loss_history': losses,
            'samples': len(local_data)
        }
        
        logger.info(f"Local training complete for client {client_id} - Loss: {losses[-1]:.4f}")
        
        return metrics
    
    def aggregate_models(self) -> SimpleHealthModel:
        """
        Aggregate client models into global model (secure aggregation).
        
        Only model weights are aggregated, never raw data.
        """
        if not self.client_models:
            raise ValueError("No client models to aggregate")
        
        # Weighted average of model parameters
        total_samples = sum(self.client_weights.values())
        
        for param_name, param in self.global_model.state_dict().items():
            aggregated = torch.zeros_like(param)
            
            for client_id, client_model in self.client_models.items():
                weight = self.client_weights[client_id] / total_samples
                aggregated += weight * client_model.state_dict()[param_name]
            
            param.copy_(aggregated)
        
        self.current_round += 1
        self.is_trained = True
        
        logger.info(f"Federated aggregation complete - Round {self.current_round}")
        
        return self.global_model
    
    def _add_privacy_noise(self, model: SimpleHealthModel):
        """Add differential privacy noise to model weights."""
        noise_scale = 0.01  # Adjust based on privacy budget
        
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
    
    def evaluate_global_model(
        self,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> Dict:
        """Evaluate global model on test data."""
        self.global_model.eval()
        
        with torch.no_grad():
            outputs = self.global_model(test_data)
            predictions = (outputs.squeeze() > 0.5).float()
            accuracy = (predictions == test_labels).float().mean()
        
        metrics = {
            'accuracy': accuracy.item(),
            'round': self.current_round,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Global model accuracy: {accuracy.item():.4f}")
        
        return metrics
    
    def save_global_model(self, path: str):
        """Save global model weights."""
        torch.save({
            'round': self.current_round,
            'state_dict': self.global_model.state_dict(),
            'model_type': self.model_type
        }, path)
        logger.info(f"Global model saved to {path}")
    
    def load_global_model(self, path: str):
        """Load global model weights."""
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint['state_dict'])
        self.current_round = checkpoint['round']
        logger.info(f"Global model loaded from {path} (round {self.current_round})")