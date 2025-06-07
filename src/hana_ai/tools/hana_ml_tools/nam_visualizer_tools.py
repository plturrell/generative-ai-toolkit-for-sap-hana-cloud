"""
This module contains visualization tools for Neural Additive Models.

The following classes are available:

    * :class `NAMFeatureContributionsPlot`
"""
import json
import logging
from typing import Optional, Type, Dict, Any, List, Union

from pydantic import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext
from hana_ml.model_storage import ModelStorage
from hana_ai.tools.hana_ml_tools.utility import _CustomEncoder
from hana_ai.tools.hana_ml_tools.neural_additive_models_tools import NeuralAdditiveModel

# Conditionally import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import base64
    import io
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Set up logging
logger = logging.getLogger(__name__)

class VisualizationInput(BaseModel):
    """
    Input schema for NAM visualization tools
    """
    name: str = Field(description="Name of the model")
    version: Optional[int] = Field(description="Model version", default=None)
    data_table: Optional[str] = Field(description="Table with sample data for visualization", default=None)
    sample_size: Optional[int] = Field(description="Number of samples to visualize", default=10)
    visualization_type: Optional[str] = Field(
        description="Type of visualization: 'feature_importance', 'contributions', 'shape_functions'", 
        default="feature_importance"
    )
    output_format: Optional[str] = Field(
        description="Output format: 'image', 'html', or 'data'", 
        default="image"
    )
    theme: Optional[str] = Field(
        description="Visual theme: 'light', 'dark', or 'colorblind'",
        default="light"
    )

class NAMFeatureContributionsPlot(BaseTool):
    r"""
    This tool creates visualizations for Neural Additive Model feature contributions and importance.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.
       
    Returns
    -------
    str
        JSON with visualization data or base64-encoded image.

        .. note::

            args_schema is used to define the schema of the inputs:

            .. list-table::
                :widths: 15 50
                :header-rows: 1

                * - Field
                  - Description
                * - name
                  - Name of the model.
                * - version
                  - Model version.
                * - data_table
                  - Table with sample data for visualization.
                * - sample_size
                  - Number of samples to visualize.
                * - visualization_type
                  - Type of visualization: 'feature_importance', 'contributions', 'shape_functions'.
                * - output_format
                  - Output format: 'image', 'html', or 'data'.
                * - theme
                  - Visual theme: 'light', 'dark', or 'colorblind'.
    """
    name: str = "nam_feature_contributions_plot"
    """Name of the tool."""
    description: str = "Create interactive visualizations for HANA Neural Additive Models to understand feature effects and importance."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = VisualizationInput
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
        data_table: Optional[str] = None,
        sample_size: Optional[int] = 10,
        visualization_type: Optional[str] = "feature_importance",
        output_format: Optional[str] = "image",
        theme: Optional[str] = "light",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Check visualization libraries availability
        if not HAS_VISUALIZATION:
            return "Visualization libraries (matplotlib, numpy, pandas) are not available."
        
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
            
            # Get visualization data based on type
            if visualization_type == "feature_importance":
                # Use feature importance from metadata if available
                feature_importance = model_meta.get('feature_importance')
                
                if not feature_importance and data_table:
                    # Load the model to calculate feature importance with data
                    model_path = ms.get_external_model_path(name=name, version=version)
                    model = NeuralAdditiveModel.load(model_path)
                    feature_importance = model.get_feature_importance()
                
                if not feature_importance:
                    return "Feature importance data is not available. Please provide a data table for calculation."
                
                # Create visualization
                return self._create_importance_visualization(
                    feature_importance=feature_importance,
                    output_format=output_format,
                    theme=theme
                )
                
            elif visualization_type == "contributions":
                # Need data table for contributions
                if not data_table:
                    return "Data table is required for feature contributions visualization."
                
                # Check if table exists
                if not self.connection_context.has_table(data_table):
                    return f"Table {data_table} does not exist in the database."
                
                # Get features from metadata
                features = model_meta.get('features', [])
                
                # Load the model
                model_path = ms.get_external_model_path(name=name, version=version)
                model = NeuralAdditiveModel.load(model_path)
                
                # Get sample data
                sample_df = self.connection_context.table(data_table).limit(sample_size).collect()
                
                # Get feature contributions
                X = sample_df[features]
                contributions = model.predict_feature_contributions(X)
                
                # Create visualization
                return self._create_contributions_visualization(
                    contributions=contributions,
                    sample_data=sample_df,
                    output_format=output_format,
                    theme=theme
                )
                
            elif visualization_type == "shape_functions":
                # Need data table for shape functions
                if not data_table:
                    return "Data table is required for shape functions visualization."
                
                # Check if table exists
                if not self.connection_context.has_table(data_table):
                    return f"Table {data_table} does not exist in the database."
                
                # Get features from metadata
                features = model_meta.get('features', [])
                
                # Load the model
                model_path = ms.get_external_model_path(name=name, version=version)
                model = NeuralAdditiveModel.load(model_path)
                
                # Get sample data
                sample_df = self.connection_context.table(data_table).collect()
                
                # Create shape functions visualization
                return self._create_shape_functions_visualization(
                    model=model,
                    sample_data=sample_df,
                    features=features,
                    output_format=output_format,
                    theme=theme
                )
                
            else:
                return f"Unknown visualization type: {visualization_type}"
            
        except Exception as e:
            return f"Error occurred: {str(e)}"

    def _create_importance_visualization(
        self,
        feature_importance: Dict[str, float],
        output_format: str = "image",
        theme: str = "light"
    ) -> str:
        """Create visualization for feature importance."""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        # Apply theme
        if theme == "dark":
            plt.style.use('dark_background')
            color = 'skyblue'
        elif theme == "colorblind":
            plt.style.use('default')
            color = '#0173B2'  # Colorblind-friendly blue
        else:  # Light theme
            plt.style.use('default')
            color = '#1f77b4'  # Default blue
        
        # Create figure with clean design
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot horizontal bars
        bars = ax.barh(features, importance, color=color, alpha=0.8, height=0.6)
        
        # Add values to the end of bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = width + 0.01
            ax.text(label_position, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                    va='center', ha='left', fontsize=9)
        
        # Clean up plot
        ax.set_title('Feature Importance', fontsize=14, pad=20)
        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Return based on output format
        if output_format == "image":
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return json.dumps({
                "type": "image",
                "format": "png",
                "data": image_base64,
                "visualization_type": "feature_importance"
            }, cls=_CustomEncoder)
            
        elif output_format == "data":
            plt.close(fig)
            # Return raw data for custom visualization
            return json.dumps({
                "type": "data",
                "features": features,
                "importance": importance,
                "visualization_type": "feature_importance"
            }, cls=_CustomEncoder)
            
        else:  # html or other
            plt.close(fig)
            # Return data for HTML visualization
            return json.dumps({
                "type": "data",
                "features": features,
                "importance": importance,
                "visualization_type": "feature_importance"
            }, cls=_CustomEncoder)

    def _create_contributions_visualization(
        self,
        contributions: Dict[str, np.ndarray],
        sample_data: pd.DataFrame,
        output_format: str = "image",
        theme: str = "light"
    ) -> str:
        """Create visualization for feature contributions to predictions."""
        # Remove bias for display
        bias = contributions.pop('bias', None)
        bias_value = bias[0] if bias is not None else 0
        
        # Sort features by average absolute contribution
        features_avg_contrib = {
            feature: np.mean(np.abs(values)) 
            for feature, values in contributions.items()
        }
        sorted_features = sorted(features_avg_contrib.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        
        # For cleaner display, limit to top 8 features
        if len(feature_names) > 8:
            top_features = feature_names[:8]
        else:
            top_features = feature_names
            
        # Prepare data for visualization
        sample_indices = range(min(10, len(sample_data)))
        
        # Apply theme
        if theme == "dark":
            plt.style.use('dark_background')
            colors = plt.cm.coolwarm
        elif theme == "colorblind":
            plt.style.use('default')
            colors = plt.cm.PiYG  # Colorblind-friendly diverging colormap
        else:  # Light theme
            plt.style.use('default')
            colors = plt.cm.coolwarm
        
        # Create figure with clean design - one row per sample
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 2*len(sample_indices)), sharex=True)
        if len(sample_indices) == 1:
            axes = [axes]  # Make sure axes is a list
        
        for i, sample_idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Extract contributions for this sample
            sample_contribs = {
                feature: contributions[feature][sample_idx] 
                for feature in top_features
            }
            
            # Sort features by absolute contribution for this sample
            sorted_sample_features = sorted(
                sample_contribs.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            features = [f[0] for f in sorted_sample_features]
            contrib_values = [f[1] for f in sorted_sample_features]
            
            # Determine colors based on contribution direction
            bar_colors = [colors(0.9) if x > 0 else colors(0.1) for x in contrib_values]
            
            # Plot horizontal bars
            bars = ax.barh(features, contrib_values, color=bar_colors, alpha=0.8, height=0.6)
            
            # Add title showing predicted value
            prediction = sum(contrib_values) + bias_value
            ax.set_title(f"Sample {sample_idx+1} - Prediction: {prediction:.3f}", fontsize=11)
            
            # Clean up plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
            
            # Add values to the end of bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_position = width + 0.05 if width > 0 else width - 0.05
                ha = 'left' if width > 0 else 'right'
                ax.text(label_position, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                        va='center', ha=ha, fontsize=8)
        
        # Add common x-label
        fig.text(0.5, 0.02, 'Feature Contribution', ha='center', va='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        # Return based on output format
        if output_format == "image":
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return json.dumps({
                "type": "image",
                "format": "png",
                "data": image_base64,
                "visualization_type": "feature_contributions",
                "bias": float(bias_value) if bias_value is not None else 0
            }, cls=_CustomEncoder)
            
        elif output_format == "data":
            plt.close(fig)
            # Return raw data for custom visualization
            return json.dumps({
                "type": "data",
                "contributions": {k: v.tolist() for k, v in contributions.items()},
                "bias": float(bias_value) if bias_value is not None else 0,
                "samples": sample_indices,
                "visualization_type": "feature_contributions"
            }, cls=_CustomEncoder)
            
        else:  # html or other
            plt.close(fig)
            # Return data for HTML visualization
            return json.dumps({
                "type": "data",
                "contributions": {k: v.tolist() for k, v in contributions.items()},
                "bias": float(bias_value) if bias_value is not None else 0,
                "samples": sample_indices,
                "visualization_type": "feature_contributions"
            }, cls=_CustomEncoder)

    def _create_shape_functions_visualization(
        self,
        model: NeuralAdditiveModel,
        sample_data: pd.DataFrame,
        features: List[str],
        output_format: str = "image",
        theme: str = "light"
    ) -> str:
        """Create visualization for feature shape functions (partial dependence)."""
        # Apply theme
        if theme == "dark":
            plt.style.use('dark_background')
            color = 'skyblue'
        elif theme == "colorblind":
            plt.style.use('default')
            color = '#0173B2'  # Colorblind-friendly blue
        else:  # Light theme
            plt.style.use('default')
            color = '#1f77b4'  # Default blue
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Select top features based on importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:min(6, len(sorted_features))]]
        
        # Calculate rows and columns for subplots
        n_cols = min(3, len(top_features))
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For each feature, create a shape function plot
        for i, feature in enumerate(top_features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get feature data
            if feature in sample_data.columns:
                feature_data = sample_data[feature].values
                
                # Create range of values for this feature
                if feature_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # For numeric features
                    x_min, x_max = np.min(feature_data), np.max(feature_data)
                    x_range = np.linspace(x_min, x_max, 100)
                    
                    # Create a dummy dataset with just this feature varying
                    X_dummy = np.zeros((len(x_range), len(features)))
                    feature_idx = features.index(feature)
                    X_dummy[:, feature_idx] = x_range
                    
                    # Get shape function values
                    with torch.no_grad():
                        # Normalize the dummy dataset
                        X_normalized = (X_dummy - model.feature_means) / model.feature_stds
                        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
                        
                        # Get contribution for this feature
                        feat = X_tensor[:, feature_idx:feature_idx+1]
                        feat_net = model.feature_nets[feature_idx]
                        feat_out = feat_net(feat)
                        shape_values = torch.sum(feat_out, dim=1).numpy()
                    
                    # Plot shape function
                    ax.plot(x_range, shape_values, color=color, linewidth=2)
                    ax.set_title(f"{feature}", fontsize=12)
                    
                    # Add horizontal line at y=0
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
                    
                    # Clean up plot
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel(feature, fontsize=10)
                    ax.set_ylabel('Contribution', fontsize=10)
                
                else:
                    # For categorical features (simplified)
                    ax.text(0.5, 0.5, f"Categorical feature: {feature}", 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f"Feature not found: {feature}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        # Add overall title
        plt.suptitle('Feature Shape Functions', fontsize=16, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Return based on output format
        if output_format == "image":
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return json.dumps({
                "type": "image",
                "format": "png",
                "data": image_base64,
                "visualization_type": "shape_functions"
            }, cls=_CustomEncoder)
            
        else:  # data or html
            plt.close(fig)
            
            # For data, we need to calculate shape functions for all features
            shape_data = {}
            
            for feature in features:
                if feature in sample_data.columns:
                    feature_data = sample_data[feature].values
                    
                    # Only handle numeric features
                    if feature_data.dtype in [np.float64, np.float32, np.int64, np.int32]:
                        # For numeric features
                        x_min, x_max = np.min(feature_data), np.max(feature_data)
                        x_range = np.linspace(x_min, x_max, 50)
                        
                        # Create a dummy dataset with just this feature varying
                        X_dummy = np.zeros((len(x_range), len(features)))
                        feature_idx = features.index(feature)
                        X_dummy[:, feature_idx] = x_range
                        
                        # Get shape function values
                        with torch.no_grad():
                            # Normalize the dummy dataset
                            X_normalized = (X_dummy - model.feature_means) / model.feature_stds
                            X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
                            
                            # Get contribution for this feature
                            feat = X_tensor[:, feature_idx:feature_idx+1]
                            feat_net = model.feature_nets[feature_idx]
                            feat_out = feat_net(feat)
                            shape_values = torch.sum(feat_out, dim=1).numpy()
                        
                        shape_data[feature] = {
                            "x_values": x_range.tolist(),
                            "y_values": shape_values.tolist(),
                            "importance": feature_importance.get(feature, 0)
                        }
            
            return json.dumps({
                "type": "data",
                "shape_functions": shape_data,
                "feature_importance": feature_importance,
                "visualization_type": "shape_functions"
            }, cls=_CustomEncoder)

    async def _arun(
        self,
        name: str,
        version: Optional[int] = None,
        data_table: Optional[str] = None,
        sample_size: Optional[int] = 10,
        visualization_type: Optional[str] = "feature_importance",
        output_format: Optional[str] = "image",
        theme: Optional[str] = "light",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(
            name=name,
            version=version,
            data_table=data_table,
            sample_size=sample_size,
            visualization_type=visualization_type,
            output_format=output_format,
            theme=theme,
            run_manager=run_manager
        )