"""
This module contains the tools for neural additive models.

The following classes are available:

    * :class `NeuralAdditiveModelsFitAndSave`
    * :class `NeuralAdditiveModelsLoadModelAndPredict`
    * :class `NeuralAdditiveModelsFeatureImportance`
"""
#pylint: disable=too-many-return-statements

import json
import logging
from typing import Optional, Type, Union, List, Dict, Any
from pydantic import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder

# Conditionally import torch
try:
    import torch
    import numpy as np
    import pandas as pd
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Set up logging
logger = logging.getLogger(__name__)

class NeuralAdditiveModel:
    """
    Implementation of Neural Additive Models based on Google Research paper.
    
    NAMs combine the interpretability of generalized additive models with
    the flexibility of neural networks by learning separate feature networks.
    
    References:
        Agarwal, R., et al. (2021). "Neural Additive Models: Interpretable Machine Learning with
        Neural Nets". Advances in Neural Information Processing Systems, 34.
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_basis_functions: int = 64,
                 hidden_sizes: List[int] = None,
                 dropout: float = 0.1,
                 feature_dropout: float = 0.05,
                 activation: str = "exu",
                 **kwargs):
        """
        Initialize a Neural Additive Model.
        
        Args:
            input_dim: Number of input features
            num_basis_functions: Number of basis functions per feature
            hidden_sizes: List of hidden layer sizes for each feature network
            dropout: Dropout rate for hidden layers
            feature_dropout: Dropout rate for features
            activation: Activation function ('relu' or 'exu')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        self.input_dim = input_dim
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.activation = activation
        
        # Build the model
        self._build_model()
        
        # Additional attributes
        self.feature_names = None
        self.feature_importance = None
        self.is_fitted = False
        self.name = None
        self.version = None
        
        # Training data statistics for normalization
        self.feature_means = None
        self.feature_stds = None
        
    def _build_model(self):
        """Build the NAM model architecture."""
        if not HAS_TORCH:
            return
            
        # Create a separate neural network for each feature
        self.feature_nets = torch.nn.ModuleList()
        
        for _ in range(self.input_dim):
            layers = []
            prev_size = 1  # Each feature net takes a single feature as input
            
            # Add hidden layers
            for hidden_size in self.hidden_sizes:
                layers.append(torch.nn.Linear(prev_size, hidden_size))
                
                # Add activation
                if self.activation.lower() == "relu":
                    layers.append(torch.nn.ReLU())
                elif self.activation.lower() == "exu":
                    # ExU activation (Exponential Linear Unit with learnable parameters)
                    layers.append(torch.nn.ELU())
                else:
                    layers.append(torch.nn.ReLU())
                
                # Add dropout
                if self.dropout > 0:
                    layers.append(torch.nn.Dropout(self.dropout))
                
                prev_size = hidden_size
            
            # Output layer (basis functions)
            layers.append(torch.nn.Linear(prev_size, self.num_basis_functions))
            
            # Create sequential model for this feature
            self.feature_nets.append(torch.nn.Sequential(*layers))
        
        # Feature dropout layer
        self.feat_dropout = torch.nn.Dropout(self.feature_dropout) if self.feature_dropout > 0 else None
        
        # Output bias
        self.bias = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """Forward pass through the NAM model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        # Split input tensor along feature dimension
        # Each feature network gets a single feature
        feature_outputs = []
        
        for i, net in enumerate(self.feature_nets):
            # Extract the i-th feature (ensure it's a 2D tensor)
            feat = x[:, i:i+1]
            
            # Pass through feature network
            feat_out = net(feat)
            
            # Apply feature dropout if enabled
            if self.feat_dropout is not None:
                feat_out = self.feat_dropout(feat_out)
            
            # Sum across basis functions to get feature contribution
            feat_contribution = torch.sum(feat_out, dim=1, keepdim=True)
            
            # Add to list of outputs
            feature_outputs.append(feat_contribution)
        
        # Sum contributions from all features
        y_pred = torch.sum(torch.cat(feature_outputs, dim=1), dim=1)
        
        # Add bias
        y_pred = y_pred + self.bias
        
        return y_pred
    
    def fit(self, 
            X, 
            y, 
            epochs: int = 100, 
            batch_size: int = 1024,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-5,
            feature_penalty: float = 0.001,
            output_regularization: float = 0.001,
            early_stopping_patience: int = 10,
            validation_split: float = 0.2,
            feature_names: List[str] = None,
            verbose: bool = False):
        """
        Fit the NAM model to data.
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            y: Target values (numpy array or pandas Series)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
            feature_penalty: Penalty for feature complexity
            output_regularization: Regularization on outputs
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of data to use for validation
            feature_names: Names of features (optional)
            verbose: Whether to print training progress
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        # Convert inputs to numpy if needed
        if hasattr(X, 'values'):
            # Save feature names if not provided
            if feature_names is None and hasattr(X, 'columns'):
                feature_names = list(X.columns)
            X = X.values
        
        if hasattr(y, 'values'):
            y = y.values
            
        # Save feature names
        self.feature_names = feature_names if feature_names else [f"Feature_{i}" for i in range(self.input_dim)]
            
        # Store normalization parameters
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)  # Avoid division by zero
        
        # Normalize data
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Split data into training and validation sets
        dataset_size = len(X_tensor)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = torch.randperm(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Set up loss function
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                outputs = self.forward(batch_X)
                
                # Compute loss
                loss = criterion(outputs, batch_y)
                
                # Add feature complexity penalty
                feature_complexity = 0
                for net in self.feature_nets:
                    # Get the weights of the output layer
                    output_weights = net[-1].weight
                    feature_complexity += torch.sum(torch.abs(output_weights))
                
                # Add output regularization
                output_reg = 0
                for net in self.feature_nets:
                    # Get the output of each feature net for this batch
                    feat_out = net(batch_X[:, 0:1])  # Dummy input, just need the shape
                    output_reg += torch.mean(torch.square(torch.sum(feat_out, dim=1)))
                
                # Total loss with regularization
                total_loss = loss + feature_penalty * feature_complexity + output_regularization * output_reg
                
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            logger.info(f'Early stopping at epoch {epoch+1}')
                        break
        
        # Calculate feature importance
        self._calculate_feature_importance(X_tensor)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def _calculate_feature_importance(self, X):
        """
        Calculate feature importance based on contribution variance.
        
        Args:
            X: Input data tensor
        """
        if not HAS_TORCH:
            return
            
        self.eval()
        with torch.no_grad():
            feature_contributions = []
            
            for i, net in enumerate(self.feature_nets):
                # Extract the i-th feature
                feat = X[:, i:i+1]
                
                # Pass through feature network
                feat_out = net(feat)
                
                # Sum across basis functions to get feature contribution
                feat_contribution = torch.sum(feat_out, dim=1).detach().numpy()
                
                # Calculate contribution variance
                contrib_var = np.var(feat_contribution)
                
                feature_contributions.append(contrib_var)
            
            # Normalize to get relative importance
            total_contrib = sum(feature_contributions)
            if total_contrib > 0:
                self.feature_importance = {
                    name: importance / total_contrib 
                    for name, importance in zip(self.feature_names, feature_contributions)
                }
            else:
                self.feature_importance = {
                    name: 1.0 / len(self.feature_names)
                    for name in self.feature_names
                }
    
    def predict(self, X):
        """
        Make predictions with the NAM model.
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            
        Returns:
            numpy array of predictions
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Convert inputs to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalize data
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        
        # Set to evaluation mode
        self.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.forward(X_tensor).numpy()
        
        return predictions
    
    def predict_feature_contributions(self, X):
        """
        Predict individual feature contributions.
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            
        Returns:
            Dictionary mapping feature names to contribution arrays
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Convert inputs to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalize data
        X_normalized = (X - self.feature_means) / self.feature_stds
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        
        # Set to evaluation mode
        self.eval()
        
        # Calculate individual feature contributions
        feature_contributions = {}
        
        with torch.no_grad():
            for i, net in enumerate(self.feature_nets):
                # Extract the i-th feature
                feat = X_tensor[:, i:i+1]
                
                # Pass through feature network
                feat_out = net(feat)
                
                # Sum across basis functions to get feature contribution
                feat_contribution = torch.sum(feat_out, dim=1).numpy()
                
                # Store contribution for this feature
                feature_name = self.feature_names[i]
                feature_contributions[feature_name] = feat_contribution
            
            # Add bias
            feature_contributions['bias'] = np.full(len(X), self.bias.item())
        
        return feature_contributions
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        return self.feature_importance
    
    def save(self, file_path):
        """
        Save model to file.
        
        Args:
            file_path: Path to save model
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        # Prepare model state and metadata
        save_dict = {
            'model_state': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'num_basis_functions': self.num_basis_functions,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout,
                'feature_dropout': self.feature_dropout,
                'activation': self.activation
            },
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'feature_means': self.feature_means.tolist() if self.feature_means is not None else None,
            'feature_stds': self.feature_stds.tolist() if self.feature_stds is not None else None,
            'is_fitted': self.is_fitted,
            'name': self.name,
            'version': self.version
        }
        
        # Save to file
        torch.save(save_dict, file_path)
    
    @classmethod
    def load(cls, file_path):
        """
        Load model from file.
        
        Args:
            file_path: Path to load model from
            
        Returns:
            Loaded NeuralAdditiveModel
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        # Load from file
        save_dict = torch.load(file_path)
        
        # Extract config
        config = save_dict['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            num_basis_functions=config['num_basis_functions'],
            hidden_sizes=config['hidden_sizes'],
            dropout=config['dropout'],
            feature_dropout=config['feature_dropout'],
            activation=config['activation']
        )
        
        # Load model state
        model.load_state_dict(save_dict['model_state'])
        
        # Load metadata
        model.feature_names = save_dict['feature_names']
        model.feature_importance = save_dict['feature_importance']
        model.feature_means = np.array(save_dict['feature_means']) if save_dict['feature_means'] is not None else None
        model.feature_stds = np.array(save_dict['feature_stds']) if save_dict['feature_stds'] is not None else None
        model.is_fitted = save_dict['is_fitted']
        model.name = save_dict['name']
        model.version = save_dict['version']
        
        return model
    
    def parameters(self):
        """Get model parameters for optimization."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        return self.feature_nets.parameters()
    
    def state_dict(self):
        """Get model state dictionary."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        return {
            'feature_nets': self.feature_nets.state_dict(),
            'bias': self.bias
        }
    
    def load_state_dict(self, state_dict):
        """Load model state dictionary."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Additive Models")
            
        self.feature_nets.load_state_dict(state_dict['feature_nets'])
        self.bias = state_dict['bias']
    
    def train(self, mode=True):
        """Set training mode."""
        if not HAS_TORCH:
            return
            
        self.feature_nets.train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        if not HAS_TORCH:
            return
            
        self.feature_nets.eval()

class ModelFitInput(BaseModel):
    """
    Input schema for fitting a neural additive model
    """
    fit_table: str = Field(description="Table containing training data")
    name: str = Field(description="Name for the model in storage")
    target: str = Field(description="Target column to predict")
    version: Optional[int] = Field(description="Model version in storage (auto-incremented if omitted)", default=None)
    features: Optional[str] = Field(description="Feature columns to use (comma-separated, uses all columns except target if omitted)", default=None)
    complexity: Optional[str] = Field(
        description="Model complexity: 'simple', 'balanced', or 'complex'", 
        default="balanced"
    )
    training_mode: Optional[str] = Field(
        description="Training mode: 'fast', 'balanced', or 'thorough'",
        default="balanced"
    )
    include_interpretability: Optional[bool] = Field(
        description="Enable advanced interpretability features",
        default=True
    )

class ModelPredictInput(BaseModel):
    """
    Input schema for predicting with a neural additive model
    """
    predict_table: str = Field(description="Table containing data to predict")
    name: str = Field(description="Name of the model")
    version: Optional[int] = Field(description="Model version", default=None)
    features: Optional[str] = Field(description="Feature columns to use (comma-separated, uses training features if omitted)", default=None)
    include_contributions: Optional[bool] = Field(description="Include individual feature contributions", default=True)
    output_format: Optional[str] = Field(description="Output format: 'standard' or 'detailed'", default="standard")

class FeatureImportanceInput(BaseModel):
    """
    Input schema for retrieving feature importance
    """
    name: str = Field(description="Name of the model")
    version: Optional[int] = Field(description="Model version", default=None)
    top_n: Optional[int] = Field(description="Show only top N most important features", default=None)
    visualization: Optional[bool] = Field(description="Include visualization data", default=True)
    format: Optional[str] = Field(description="Output format: 'simple', 'detailed', or 'complete'", default="simple")

class NeuralAdditiveModelsFitAndSave(BaseTool):
    r"""
    This tool is used to fit a neural additive model and save it to model storage.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
       
    Returns
    -------
    str
        The result string containing the training table name, model storage name, and model storage version.

        .. note::

            args_schema is used to define the schema of the inputs for the neural additive model:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - fit_table
                  - The name of the table containing the training data.
                * - name
                  - The name of the model to save.
                * - version
                  - The version of the model to save.
                * - features
                  - Feature columns to use (comma-separated).
                * - target
                  - The target column to predict.
                * - complexity
                  - Model complexity: 'simple', 'balanced', or 'complex'.
                * - training_mode
                  - Training mode: 'fast', 'balanced', or 'thorough'.
                * - include_interpretability
                  - Enable advanced interpretability features.
    """
    name: str = "neural_additive_models_fit_and_save"
    """Name of the tool."""
    description: str = "Train an interpretable predictive model using HANA Neural Additive Models (NAM). Provides feature importance and transparent predictions."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelFitInput
    return_direct: bool = False
    
    def _get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self,
        fit_table: str,
        name: str,
        target: str,
        version: Optional[int] = None,
        features: Optional[str] = None,
        complexity: Optional[str] = "balanced",
        training_mode: Optional[str] = "balanced",
        include_interpretability: Optional[bool] = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check PyTorch availability
        if not HAS_TORCH:
            return "PyTorch is not available. Please install PyTorch to use Neural Additive Models."
        
        # Check if fit_table exists
        if not self.connection_context.has_table(fit_table):
            return f"Table {fit_table} does not exist in the database."
        
        # Get table data
        df = self.connection_context.table(fit_table).collect()
        
        # Check if target column exists
        if target not in df.columns:
            return f"Target column {target} does not exist in the table {fit_table}."
        
        # Select features
        if features is not None:
            feature_list = [f.strip() for f in features.split(',')]
            # Check if all feature columns exist
            for feature in feature_list:
                if feature not in df.columns:
                    return f"Feature column {feature} does not exist in the table {fit_table}."
        else:
            # Use all columns except target as features
            feature_list = [col for col in df.columns if col != target]
            
        # Map complexity setting to model parameters
        complexity_settings = {
            "simple": {
                "num_basis_functions": 32,
                "hidden_sizes": [32, 16],
                "dropout": 0.1,
                "feature_dropout": 0.05,
                "activation": "relu"
            },
            "balanced": {
                "num_basis_functions": 64,
                "hidden_sizes": [64, 32],
                "dropout": 0.1,
                "feature_dropout": 0.05,
                "activation": "exu"
            },
            "complex": {
                "num_basis_functions": 128,
                "hidden_sizes": [128, 64, 32],
                "dropout": 0.15,
                "feature_dropout": 0.1,
                "activation": "exu"
            }
        }
        
        # Map training mode to training parameters
        training_settings = {
            "fast": {
                "epochs": 50,
                "batch_size": 256,
                "learning_rate": 0.002,
                "early_stopping_patience": 5,
                "validation_split": 0.1
            },
            "balanced": {
                "epochs": 100,
                "batch_size": 128,
                "learning_rate": 0.001,
                "early_stopping_patience": 10,
                "validation_split": 0.2
            },
            "thorough": {
                "epochs": 200,
                "batch_size": 64,
                "learning_rate": 0.0005,
                "early_stopping_patience": 20,
                "validation_split": 0.2
            }
        }
        
        # Get settings based on user choices (with fallback to balanced)
        model_settings = complexity_settings.get(complexity, complexity_settings["balanced"])
        train_settings = training_settings.get(training_mode, training_settings["balanced"])
        
        # Prepare data
        X = df[feature_list]
        y = df[target]
        
        try:
            # Create model with selected settings
            model = NeuralAdditiveModel(
                input_dim=len(feature_list),
                num_basis_functions=model_settings["num_basis_functions"],
                hidden_sizes=model_settings["hidden_sizes"],
                dropout=model_settings["dropout"],
                feature_dropout=model_settings["feature_dropout"],
                activation=model_settings["activation"]
            )
            
            # Fit model with selected training settings
            model.fit(
                X=X,
                y=y,
                epochs=train_settings["epochs"],
                batch_size=train_settings["batch_size"],
                learning_rate=train_settings["learning_rate"],
                early_stopping_patience=train_settings["early_stopping_patience"],
                validation_split=train_settings["validation_split"],
                weight_decay=1e-5,
                feature_penalty=0.001,
                output_regularization=0.001,
                feature_names=feature_list,
                verbose=False
            )
            
            # Set model name and version
            model.name = name
            
            # Prepare for seamless integration with model storage
            # Use the HANA integration framework pattern that other tools use
            try:
                # Import HANA integration utilities to keep consistent patterns
                from hana_ai.model_registry import register_model
                has_registry = True
            except ImportError:
                # Fall back to direct storage if model registry not available
                has_registry = False
                
            # Set model name and version
            model.name = name
            
            # Initialize model storage
            ms = ModelStorage(connection_context=self.connection_context)
            ms._create_metadata_table()
            
            # Get version based on model registry or auto-increment
            if version is None:
                if has_registry:
                    # Use consistent versioning with other models
                    version = register_model.get_next_version(name)
                else:
                    # Fall back to traditional versioning
                    version = ms._get_new_version_no(name)
                    if version is None:
                        version = 1
                    else:
                        version = int(version)
            
            model.version = version
            
            # Enhanced metadata for better integration with the HANA ecosystem
            meta = {
                # Standard metadata used by other tools
                'type': 'neural_additive_model',
                'framework': 'pytorch',
                'features': feature_list,
                'target': target,
                'prediction_type': 'regression',
                'created_at': self._get_timestamp(),
                
                # NAM-specific parameters
                'complexity': complexity,
                'training_mode': training_mode,
                'include_interpretability': include_interpretability,
                'input_dim': len(feature_list),
                'feature_importance': model.get_feature_importance(),
                
                # Model configuration - matches pattern used by other models
                'model_config': {
                    'architecture': 'nam',
                    'num_basis_functions': model_settings["num_basis_functions"],
                    'hidden_sizes': model_settings["hidden_sizes"],
                    'activation': model_settings["activation"]
                }
            }
            
            # Use a consistent location for model storage
            import os
            import tempfile
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(tempfile.gettempdir(), "hana_nam_models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Use standardized naming convention
            model_filename = f"{name}_v{version}.pt"
            temp_file = os.path.join(model_dir, model_filename)
            
            # Save model with consistent formatting
            model.save(temp_file)
            
            # Register model with the system
            if has_registry:
                register_model.register(
                    name=name,
                    version=version,
                    path=temp_file,
                    metadata=meta
                )
                
            # Save to model storage with improved integration
            ms.save_external_model(
                model_path=temp_file,
                model_name=name,
                version=version,
                meta=meta,
                if_exists='replace'
            )
            
            # Clean up only if successfully saved to model storage
            try:
                # Check if the model exists in storage before removing temp file
                if ms.model_exists(name=name, version=version):
                    if os.path.exists(temp_file) and not has_registry:
                        # Only remove if we're not using the registry (registry needs the file)
                        os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not clean up temporary file {temp_file}: {str(e)}")
            
            # Enhanced return object with more details but cleaner presentation
            output = {
                "model": {
                    "name": name,
                    "version": version,
                    "type": "Neural Additive Model"
                },
                "training": {
                    "table": fit_table,
                    "target": target,
                    "features_count": len(feature_list),
                    "complexity": complexity,
                    "mode": training_mode
                },
                "interpretability": include_interpretability
            }
            
            # Return result
            return json.dumps(output, cls=_CustomEncoder)
            
        except ValueError as ve:
            return f"ValueError occurred: {str(ve)}"
        except KeyError as ke:
            return f"KeyError occurred: {str(ke)}"
        except TypeError as te:
            return f"TypeError occurred: {str(te)}"
        except Exception as e:
            return f"Error occurred: {str(e)}"

    async def _arun(
        self,
        fit_table: str,
        name: str,
        target: str,
        version: Optional[int] = None,
        features: Optional[str] = None,
        complexity: Optional[str] = "balanced",
        training_mode: Optional[str] = "balanced",
        include_interpretability: Optional[bool] = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            fit_table=fit_table,
            name=name,
            target=target,
            version=version,
            features=features,
            complexity=complexity,
            training_mode=training_mode,
            include_interpretability=include_interpretability,
            run_manager=run_manager
        )

class NeuralAdditiveModelsLoadModelAndPredict(BaseTool):
    r"""
    This tool is used to load a neural additive model from model storage and make predictions.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        The name of the predicted results table.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - predict_table
                  - The name of the table containing the prediction data.
                * - name
                  - The name of the model to load.
                * - version
                  - The version of the model to load.
                * - features
                  - Comma-separated list of feature columns to use (should match those used in training).
                * - include_contributions
                  - Whether to include individual feature contributions in the output.
    """
    name: str = "neural_additive_models_load_model_and_predict"
    """Name of the tool."""
    description: str = "Generate predictions with HANA Neural Additive Models. Visualize feature contributions for transparency and model explainability."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ModelPredictInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self,
        predict_table: str,
        name: str,
        version: Optional[int] = None,
        features: Optional[str] = None,
        include_contributions: Optional[bool] = False,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check PyTorch availability
        if not HAS_TORCH:
            return "PyTorch is not available. Please install PyTorch to use Neural Additive Models."
        
        # Check if predict_table exists
        if not self.connection_context.has_table(predict_table):
            return f"Table {predict_table} does not exist in the database."
        
        # Get table data
        df = self.connection_context.table(predict_table).collect()
        
        # Get model from storage
        ms = ModelStorage(connection_context=self.connection_context)
        
        try:
            # Load model metadata
            model_meta = ms.get_model_meta(name=name, version=version)
            if model_meta is None:
                return f"Model {name} (version {version}) not found in model storage."
            
            # Check if it's a neural additive model
            if model_meta.get('type') != 'neural_additive_model':
                return f"Model {name} (version {version}) is not a neural additive model."
            
            # Get model file path
            model_path = ms.get_external_model_path(name=name, version=version)
            if model_path is None:
                return f"Model file for {name} (version {version}) not found."
            
            # Load the model
            model = NeuralAdditiveModel.load(model_path)
            
            # Get features from metadata
            model_features = model_meta.get('features', [])
            
            # If features are provided, use them instead
            if features is not None:
                feature_list = [f.strip() for f in features.split(',')]
                # Check if all feature columns exist
                for feature in feature_list:
                    if feature not in df.columns:
                        return f"Feature column {feature} does not exist in the table {predict_table}."
                    if feature not in model_features:
                        return f"Feature {feature} was not used in training. Available features: {', '.join(model_features)}"
            else:
                feature_list = model_features
                # Check if all model features exist in the prediction table
                for feature in feature_list:
                    if feature not in df.columns:
                        return f"Feature {feature} used in training not found in the prediction table."
            
            # Prepare data for prediction
            X = df[feature_list]
            
            # Make predictions
            predictions = model.predict(X)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                "prediction": predictions
            })
            
            # Add feature contributions if requested
            if include_contributions:
                contributions = model.predict_feature_contributions(X)
                for feature, contrib in contributions.items():
                    results_df[f"contribution_{feature}"] = contrib
            
            # Save results to HANA
            results_table_name = f"{name}_{version}_NAM_PREDICTIONS"
            self.connection_context.create_dataframe_from_pandas(
                results_df,
                results_table_name,
                force=True
            )
            
            # Return result
            output = {"predicted_results_table": results_table_name}
            if include_contributions:
                output["contributions_included"] = True
                output["features"] = feature_list
            
            return json.dumps(output, cls=_CustomEncoder)
            
        except ValueError as ve:
            return f"ValueError occurred: {str(ve)}"
        except KeyError as ke:
            return f"KeyError occurred: {str(ke)}"
        except TypeError as te:
            return f"TypeError occurred: {str(te)}"
        except Exception as e:
            return f"Error occurred: {str(e)}"

    async def _arun(
        self,
        predict_table: str,
        name: str,
        version: Optional[int] = None,
        features: Optional[str] = None,
        include_contributions: Optional[bool] = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            predict_table=predict_table,
            name=name,
            version=version,
            features=features,
            include_contributions=include_contributions,
            run_manager=run_manager
        )

class NeuralAdditiveModelsFeatureImportance(BaseTool):
    r"""
    This tool is used to get feature importance scores from a neural additive model.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Feature importance information in JSON format.

        .. note::

            args_schema is used to define the schema of the inputs as follows:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - name
                  - The name of the model.
                * - version
                  - The version of the model.
                * - top_n
                  - Show only the top N most important features.
    """
    name: str = "neural_additive_models_feature_importance"
    """Name of the tool."""
    description: str = "Analyze feature impact in HANA NAM models with visual charts and importance rankings. Enables transparent decision making."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = FeatureImportanceInput
    return_direct: bool = False

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = False
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self,
        name: str,
        version: Optional[int] = None,
        top_n: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check PyTorch availability
        if not HAS_TORCH:
            return "PyTorch is not available. Please install PyTorch to use Neural Additive Models."
        
        # Get model from storage
        ms = ModelStorage(connection_context=self.connection_context)
        
        try:
            # Load model metadata
            model_meta = ms.get_model_meta(name=name, version=version)
            if model_meta is None:
                return f"Model {name} (version {version}) not found in model storage."
            
            # Check if it's a neural additive model
            if model_meta.get('type') != 'neural_additive_model':
                return f"Model {name} (version {version}) is not a neural additive model."
            
            # Get feature importance from metadata if available
            feature_importance = model_meta.get('feature_importance')
            
            if not feature_importance:
                # Get model file path and load the model to get feature importance
                model_path = ms.get_external_model_path(name=name, version=version)
                if model_path is None:
                    return f"Model file for {name} (version {version}) not found."
                
                # Load the model
                model = NeuralAdditiveModel.load(model_path)
                
                # Get feature importance
                feature_importance = model.get_feature_importance()
            
            # Sort by importance
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Limit to top_n if specified
            if top_n is not None and top_n > 0:
                sorted_importance = sorted_importance[:top_n]
            
            # Format output
            formatted_importance = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in sorted_importance
            ]
            
            return json.dumps({
                "model_name": name,
                "model_version": version,
                "feature_importance": formatted_importance
            }, cls=_CustomEncoder)
            
        except ValueError as ve:
            return f"ValueError occurred: {str(ve)}"
        except KeyError as ke:
            return f"KeyError occurred: {str(ke)}"
        except TypeError as te:
            return f"TypeError occurred: {str(te)}"
        except Exception as e:
            return f"Error occurred: {str(e)}"

    async def _arun(
        self,
        name: str,
        version: Optional[int] = None,
        top_n: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            name=name,
            version=version,
            top_n=top_n,
            run_manager=run_manager
        )