"""
Neural Additive Models Design System

A comprehensive design system for the Neural Additive Models integration
within the HANA AI Toolkit. This package provides React-based UI components
and Python integration to create a cohesive, emotionally resonant experience
for users interacting with Neural Additive Models.

Components:
- ComplexitySelector: A precision-crafted control for selecting model complexity
- FeatureContributions: A visualization component for feature contributions
- TrainingVisualizer: A visualization of the training process
- ModelExplorer: An exploration tool for model behavior across different feature values

Integration:
- Integration module for connecting with Python backend
"""

from hana_ai.tools.hana_ml_tools.nam_design_system.integration import (
    create_nam_integration,
    NAMDesignSystemIntegration
)

__all__ = [
    'create_nam_integration',
    'NAMDesignSystemIntegration'
]