"""
This module contains the functions to fetch data from HANA.

The following classes are available:

    * :class `DummyTool`
"""

import logging
from typing import Optional, Type, List, Dict, Union, Any
from pydantic import BaseModel, Field, validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

from hana_ml import ConnectionContext

logger = logging.getLogger(__name__)

class ClassificationToolInput(BaseModel):
    """
    Input schema for the ClassificationTool.
    """
    table_name: str = Field(description="The name of the table containing data for training or prediction.")
    target_column: str = Field(description="The name of the column to predict (label column).")
    feature_columns: Optional[Union[List[str], str]] = Field(
        description="Column names to use as features. If not provided, all columns except target will be used.", 
        default=None
    )
    mode: str = Field(
        description="'train' to train a new model, 'predict' to make predictions with an existing model.", 
        default="predict"
    )
    model_name: Optional[str] = Field(
        description="Name to identify the model. Will be used to store or retrieve the model.", 
        default=None
    )
    algorithm: Optional[str] = Field(
        description="Algorithm to use for classification. Options: 'RandomForest', 'GradientBoosting', 'SVM', 'LogisticRegression'.", 
        default="RandomForest"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        description="Optional hyperparameters for the selected algorithm.", 
        default=None
    )
    
    @validator('mode')
    def validate_mode(cls, v):
        if v.lower() not in ['train', 'predict']:
            raise ValueError("Mode must be either 'train' or 'predict'")
        return v.lower()
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['RandomForest', 'GradientBoosting', 'SVM', 'LogisticRegression']
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        return v


class RegressionToolInput(BaseModel):
    """
    Input schema for the RegressionTool.
    """
    table_name: str = Field(description="The name of the table containing data for training or prediction.")
    target_column: str = Field(description="The name of the column to predict (target column).")
    feature_columns: Optional[Union[List[str], str]] = Field(
        description="Column names to use as features. If not provided, all columns except target will be used.", 
        default=None
    )
    mode: str = Field(
        description="'train' to train a new model, 'predict' to make predictions with an existing model.", 
        default="predict"
    )
    model_name: Optional[str] = Field(
        description="Name to identify the model. Will be used to store or retrieve the model.", 
        default=None
    )
    algorithm: Optional[str] = Field(
        description="Algorithm to use for regression. Options: 'GradientBoosting', 'RandomForest', 'LinearRegression'.", 
        default="GradientBoosting"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        description="Optional hyperparameters for the selected algorithm.", 
        default=None
    )
    
    @validator('mode')
    def validate_mode(cls, v):
        if v.lower() not in ['train', 'predict']:
            raise ValueError("Mode must be either 'train' or 'predict'")
        return v.lower()
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        valid_algorithms = ['GradientBoosting', 'RandomForest', 'LinearRegression']
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm must be one of {valid_algorithms}")
        return v

class ClassificationTool(BaseTool):
    """
    Tool for training and using classification models with SAP HANA.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Result of the classification operation.
    """
    name: str = "classification_tool"
    """Name of the tool."""
    description: str = "Trains or runs predictions with a classification model on SAP HANA data. Requires table_name, target_column, and optionally feature_columns, mode (train/predict), and model_name."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = ClassificationToolInput
    """Input schema of the tool."""
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = True
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool for classification tasks."""
        try:
            # Extract parameters
            table_name = kwargs.get("table_name")
            target_column = kwargs.get("target_column")
            feature_columns = kwargs.get("feature_columns")
            training_mode = kwargs.get("mode", "predict").lower() == "train"
            model_name = kwargs.get("model_name", "classification_model")
            
            if not table_name or not target_column:
                return "Error: table_name and target_column are required parameters."
                
            # Implement actual classification functionality using SAP HANA PAL
            from hana_ml.algorithms.pal.unified_classification import UnifiedClassification
            
            # Validate table exists
            conn = self.connection_context
            try:
                df = conn.table(table_name)
            except Exception as e:
                return f"Error: Could not access table {table_name}. {str(e)}"
            
            # Create feature list from feature_columns or use all columns except target
            if feature_columns:
                features = feature_columns if isinstance(feature_columns, list) else [feature_columns]
            else:
                # Get all columns from the table
                table_columns = df.columns
                features = [col for col in table_columns if col != target_column]
            
            if training_mode:
                # Training mode - create and train a classification model
                try:
                    # Create classifier with default parameters
                    classifier = UnifiedClassification(
                        conn_context=conn,
                        algorithm='RandomForest',
                        model_name=model_name,
                        n_estimators=100,
                        max_depth=10,
                        thread_ratio=1.0
                    )
                    
                    # Fit the model
                    classifier.fit(data=df, key=None, features=features, label=target_column)
                    
                    # Get model statistics
                    model_stats = classifier.get_model_info()
                    accuracy = classifier.score(df, target_column)
                    
                    return f"Classification model '{model_name}' successfully trained on table {table_name}. Model accuracy: {accuracy:.4f}"
                    
                except Exception as e:
                    logger.error(f"Classification training error: {str(e)}")
                    return f"Error training classification model: {str(e)}"
            else:
                # Prediction mode - load model and make predictions
                try:
                    # Load the model
                    classifier = UnifiedClassification(
                        conn_context=conn,
                        algorithm='RandomForest',
                        model_name=model_name
                    )
                    
                    # Make predictions
                    predictions = classifier.predict(df)
                    
                    # Get prediction summary
                    summary = predictions.collect().head(5).to_dict()
                    
                    return f"Made predictions using model '{model_name}' on table {table_name}. First 5 predictions: {summary}"
                    
                except Exception as e:
                    logger.error(f"Classification prediction error: {str(e)}")
                    return f"Error making predictions with classification model: {str(e)}"
                    
        except Exception as e:
            logger.error(f"Classification tool error: {str(e)}")
            return f"Error in classification tool: {str(e)}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(run_manager=run_manager)

class RegressionTool(BaseTool):
    """
    Tool for training and using regression models with SAP HANA.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection context to the HANA database.

    Returns
    -------
    str
        Result of the regression operation.
    """
    name: str = "regression_tool"
    """Name of the tool."""
    description: str = "Trains or runs predictions with a regression model on SAP HANA data. Requires table_name, target_column, and optionally feature_columns, mode (train/predict), and model_name."
    """Description of the tool."""
    connection_context: ConnectionContext = None
    """Connection context to the HANA database."""
    args_schema: Type[BaseModel] = RegressionToolInput
    """Input schema of the tool."""
    return_direct: bool = True

    def __init__(
        self,
        connection_context: ConnectionContext,
        return_direct: bool = True
    ) -> None:
        super().__init__(  # type: ignore[call-arg]
            connection_context=connection_context,
            return_direct=return_direct
        )

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool for regression tasks."""
        try:
            # Extract parameters
            table_name = kwargs.get("table_name")
            target_column = kwargs.get("target_column")
            feature_columns = kwargs.get("feature_columns")
            training_mode = kwargs.get("mode", "predict").lower() == "train"
            model_name = kwargs.get("model_name", "regression_model")
            
            if not table_name or not target_column:
                return "Error: table_name and target_column are required parameters."
                
            # Implement actual regression functionality using SAP HANA PAL
            from hana_ml.algorithms.pal.unified_regression import UnifiedRegression
            
            # Validate table exists
            conn = self.connection_context
            try:
                df = conn.table(table_name)
            except Exception as e:
                return f"Error: Could not access table {table_name}. {str(e)}"
            
            # Create feature list from feature_columns or use all columns except target
            if feature_columns:
                features = feature_columns if isinstance(feature_columns, list) else [feature_columns]
            else:
                # Get all columns from the table
                table_columns = df.columns
                features = [col for col in table_columns if col != target_column]
            
            if training_mode:
                # Training mode - create and train a regression model
                try:
                    # Create regressor with default parameters
                    regressor = UnifiedRegression(
                        conn_context=conn,
                        algorithm='GradientBoosting',
                        model_name=model_name,
                        n_estimators=100,
                        learning_rate=0.1,
                        thread_ratio=1.0
                    )
                    
                    # Fit the model
                    regressor.fit(data=df, key=None, features=features, label=target_column)
                    
                    # Get model statistics
                    model_stats = regressor.get_model_info()
                    r2_score = regressor.score(df, target_column)
                    
                    return f"Regression model '{model_name}' successfully trained on table {table_name}. Model R² score: {r2_score:.4f}"
                    
                except Exception as e:
                    logger.error(f"Regression training error: {str(e)}")
                    return f"Error training regression model: {str(e)}"
            else:
                # Prediction mode - load model and make predictions
                try:
                    # Load the model
                    regressor = UnifiedRegression(
                        conn_context=conn,
                        algorithm='GradientBoosting',
                        model_name=model_name
                    )
                    
                    # Make predictions
                    predictions = regressor.predict(df)
                    
                    # Get prediction statistics
                    stats = predictions.select_statement(f"AVG({target_column}_PRED), MIN({target_column}_PRED), MAX({target_column}_PRED)").collect()
                    
                    # Get sample predictions
                    sample = predictions.collect().head(5).to_dict()
                    
                    return f"Made predictions using model '{model_name}' on table {table_name}. Statistics: {stats}. First 5 predictions: {sample}"
                    
                except Exception as e:
                    logger.error(f"Regression prediction error: {str(e)}")
                    return f"Error making predictions with regression model: {str(e)}"
                    
        except Exception as e:
            logger.error(f"Regression tool error: {str(e)}")
            return f"Error in regression tool: {str(e)}"

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs: str
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(run_manager=run_manager)
