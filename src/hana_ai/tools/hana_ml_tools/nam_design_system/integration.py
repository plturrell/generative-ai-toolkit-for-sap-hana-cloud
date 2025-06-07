"""
Neural Additive Models Design System Integration

This module provides integration between the NAM Python backend and the React-based
design system frontend. It serves as the bridge between the two, handling data
transformation, API routing, and state management.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import NAM backend components
from hana_ai.tools.hana_ml_tools.neural_additive_models_tools import (
    NeuralAdditiveModel,
    NeuralAdditiveModelsFitAndSave,
    NeuralAdditiveModelsLoadModelAndPredict,
    NeuralAdditiveModelsFeatureImportance
)
from hana_ai.tools.hana_ml_tools.nam_visualizer_tools import NAMFeatureContributionsPlot

# Path to design system static files
DESIGN_SYSTEM_PATH = Path(__file__).parent

# Models for request/response schema
class TrainingRequest(BaseModel):
    """Request schema for model training"""
    fit_table: str
    name: str
    target: str
    version: Optional[int] = None
    features: Optional[str] = None
    complexity: str = "balanced"
    training_mode: str = "balanced"
    include_interpretability: bool = True

class PredictionRequest(BaseModel):
    """Request schema for model prediction"""
    predict_table: str
    name: str
    version: Optional[int] = None
    features: Optional[str] = None
    include_contributions: bool = True
    output_format: str = "standard"

class FeatureImportanceRequest(BaseModel):
    """Request schema for feature importance"""
    name: str
    version: Optional[int] = None
    top_n: Optional[int] = None
    visualization: bool = True
    format: str = "simple"

class ShapeFunctionRequest(BaseModel):
    """Request schema for shape functions"""
    name: str
    version: Optional[int] = None
    data_table: str
    features: Optional[List[str]] = None
    output_format: str = "data"

class NAMDesignSystemIntegration:
    """
    Integration class for Neural Additive Models Design System
    
    This class provides the necessary APIs and methods to connect the React-based
    frontend design system with the Python backend implementation of Neural
    Additive Models.
    """
    
    def __init__(self, connection_context=None):
        """Initialize the integration with a database connection context"""
        self.connection_context = connection_context
        self.router = APIRouter(prefix="/api/nam", tags=["nam"])
        self._register_routes()
        
    def _register_routes(self):
        """Register API routes"""
        # Training endpoints
        self.router.post("/train", response_model=Dict[str, Any])(self.train_model)
        
        # Prediction endpoints
        self.router.post("/predict", response_model=Dict[str, Any])(self.predict)
        
        # Feature importance endpoints
        self.router.get("/feature-importance", response_model=Dict[str, Any])(self.get_feature_importance)
        
        # Shape function endpoints
        self.router.post("/shape-functions", response_model=Dict[str, Any])(self.get_shape_functions)
        
        # UI endpoints
        self.router.get("/ui", response_class=HTMLResponse)(self.get_ui)
        
    async def train_model(self, request: TrainingRequest) -> Dict[str, Any]:
        """Train a Neural Additive Model"""
        if not self.connection_context:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        try:
            # Create NAM training tool
            nam_tool = NeuralAdditiveModelsFitAndSave(connection_context=self.connection_context)
            
            # Execute training
            result = nam_tool._run(
                fit_table=request.fit_table,
                name=request.name,
                target=request.target,
                version=request.version,
                features=request.features,
                complexity=request.complexity,
                training_mode=request.training_mode,
                include_interpretability=request.include_interpretability
            )
            
            # Parse result (which is JSON string) back to dict
            return json.loads(result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
    
    async def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make predictions with a Neural Additive Model"""
        if not self.connection_context:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        try:
            # Create NAM prediction tool
            nam_tool = NeuralAdditiveModelsLoadModelAndPredict(connection_context=self.connection_context)
            
            # Execute prediction
            result = nam_tool._run(
                predict_table=request.predict_table,
                name=request.name,
                version=request.version,
                features=request.features,
                include_contributions=request.include_contributions
            )
            
            # Parse result
            prediction_result = json.loads(result)
            
            # If we have a prediction table with contributions, enhance the response
            # with visualization data for the frontend
            if request.include_contributions and "contributions_included" in prediction_result:
                # In a real implementation, we would fetch the prediction data
                # and format it for the frontend components
                prediction_result["sample_predictions"] = self._get_sample_predictions(
                    prediction_result["predicted_results_table"], 
                    limit=10
                )
                
            return prediction_result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
    
    async def get_feature_importance(self, name: str, version: Optional[int] = None, top_n: Optional[int] = None) -> Dict[str, Any]:
        """Get feature importance for a Neural Additive Model"""
        if not self.connection_context:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        try:
            # Create NAM feature importance tool
            nam_tool = NeuralAdditiveModelsFeatureImportance(connection_context=self.connection_context)
            
            # Execute feature importance calculation
            result = nam_tool._run(
                name=name,
                version=version,
                top_n=top_n
            )
            
            # Parse result
            return json.loads(result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")
    
    async def get_shape_functions(self, request: ShapeFunctionRequest) -> Dict[str, Any]:
        """Get shape functions for a Neural Additive Model"""
        if not self.connection_context:
            raise HTTPException(status_code=500, detail="Database connection not available")
            
        try:
            # Create NAM visualization tool
            nam_tool = NAMFeatureContributionsPlot(connection_context=self.connection_context)
            
            # Execute shape function visualization
            result = nam_tool._run(
                name=request.name,
                version=request.version,
                data_table=request.data_table,
                visualization_type="shape_functions",
                output_format=request.output_format
            )
            
            # Parse result
            return json.loads(result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting shape functions: {str(e)}")
    
    async def get_ui(self, request: Request) -> HTMLResponse:
        """Return the HTML for the NAM Design System UI"""
        # Simple HTML template that loads the NAM design system
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Neural Additive Models Design System</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f2f6fa;
                }
                #nam-app {
                    margin: 2rem auto;
                    max-width: 1200px;
                }
            </style>
        </head>
        <body>
            <div id="nam-app"></div>
            
            <script type="module">
                // Import the NAM design system
                import { initNAMApp } from '/static/nam/index.js';
                
                // Initialize the app
                window.addEventListener('DOMContentLoaded', () => {
                    initNAMApp('nam-app', {
                        apiEndpoint: '/api/nam',
                        theme: 'light'
                    });
                });
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    
    def _get_sample_predictions(self, table_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches sample predictions from the specified table
        
        In a real implementation, this would execute a database query.
        For this example, we're returning simulated data.
        """
        # Simulated data - in a real implementation, this would query the database
        return [
            {
                "id": i,
                "prediction": 10.5 + i * 0.75,
                "contributions": {
                    "PRICE": 2.5 + (i % 3) * 0.25,
                    "QUANTITY": 1.2 - (i % 2) * 0.3,
                    "REGION": 0.8 + (i % 4) * 0.1,
                    "CUSTOMER_SEGMENT": 1.5 + (i % 3) * 0.5
                }
            } for i in range(limit)
        ]
    
    def mount_static_files(self, app):
        """Mount static files for the design system"""
        static_dir = DESIGN_SYSTEM_PATH
        app.mount("/static/nam", StaticFiles(directory=static_dir), name="nam_static")

def create_nam_integration(connection_context=None):
    """
    Create a Neural Additive Models Design System integration
    
    Args:
        connection_context: HANA database connection context
        
    Returns:
        NAMDesignSystemIntegration instance
    """
    return NAMDesignSystemIntegration(connection_context=connection_context)