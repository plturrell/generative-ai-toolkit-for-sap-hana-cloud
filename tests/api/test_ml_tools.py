"""
Unit tests for the machine learning tools implementations.
"""
import unittest
from unittest.mock import MagicMock, patch
import pytest

from hana_ai.tools.hana_ml_tools.unsupported_tools import (
    ClassificationTool,
    RegressionTool,
    ClassificationToolInput,
    RegressionToolInput
)


class TestClassificationTool(unittest.TestCase):
    """Tests for the ClassificationTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.mock_df = MagicMock()
        self.mock_conn.table.return_value = self.mock_df
        self.mock_df.columns = ["feature1", "feature2", "target"]
        
        self.tool = ClassificationTool(connection_context=self.mock_conn)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedClassification")
    def test_classification_training_mode(self, mock_classifier_class):
        """Test classification tool in training mode."""
        # Setup mock classifier
        mock_classifier = MagicMock()
        mock_classifier_class.return_value = mock_classifier
        mock_classifier.score.return_value = 0.85
        
        # Call the tool in training mode
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            mode="train",
            model_name="test_model"
        )
        
        # Verify expectations
        self.mock_conn.table.assert_called_once_with("TEST_TABLE")
        mock_classifier_class.assert_called_once()
        mock_classifier.fit.assert_called_once()
        self.assertIn("successfully trained", result)
        self.assertIn("0.8500", result)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedClassification")
    def test_classification_prediction_mode(self, mock_classifier_class):
        """Test classification tool in prediction mode."""
        # Setup mock classifier and predictions
        mock_classifier = MagicMock()
        mock_classifier_class.return_value = mock_classifier
        
        mock_predictions = MagicMock()
        mock_classifier.predict.return_value = mock_predictions
        
        # Mock the collection of predictions
        mock_predictions.collect.return_value.head.return_value.to_dict.return_value = {
            "id": [1, 2, 3],
            "target_PRED": ["class1", "class2", "class1"]
        }
        
        # Call the tool in prediction mode
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            model_name="test_model"
        )
        
        # Verify expectations
        mock_classifier_class.assert_called_once()
        mock_classifier.predict.assert_called_once()
        self.assertIn("Made predictions", result)
        self.assertIn("test_model", result)
    
    def test_classification_missing_parameters(self):
        """Test classification tool with missing parameters."""
        # Call without required params
        result = self.tool._run(target_column="target")
        self.assertIn("Error", result)
        
        result = self.tool._run(table_name="TEST_TABLE")
        self.assertIn("Error", result)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedClassification")
    def test_classification_error_handling(self, mock_classifier_class):
        """Test classification tool error handling."""
        # Setup mock to raise exception
        mock_classifier_class.side_effect = ValueError("Test error")
        
        # Call the tool
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            mode="train"
        )
        
        # Verify error is handled
        self.assertIn("Error", result)
        self.assertIn("Test error", result)


class TestRegressionTool(unittest.TestCase):
    """Tests for the RegressionTool."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = MagicMock()
        self.mock_df = MagicMock()
        self.mock_conn.table.return_value = self.mock_df
        self.mock_df.columns = ["feature1", "feature2", "target"]
        
        self.tool = RegressionTool(connection_context=self.mock_conn)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedRegression")
    def test_regression_training_mode(self, mock_regressor_class):
        """Test regression tool in training mode."""
        # Setup mock regressor
        mock_regressor = MagicMock()
        mock_regressor_class.return_value = mock_regressor
        mock_regressor.score.return_value = 0.78
        
        # Call the tool in training mode
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            mode="train",
            model_name="test_model"
        )
        
        # Verify expectations
        self.mock_conn.table.assert_called_once_with("TEST_TABLE")
        mock_regressor_class.assert_called_once()
        mock_regressor.fit.assert_called_once()
        self.assertIn("successfully trained", result)
        self.assertIn("0.7800", result)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedRegression")
    def test_regression_prediction_mode(self, mock_regressor_class):
        """Test regression tool in prediction mode."""
        # Setup mock regressor and predictions
        mock_regressor = MagicMock()
        mock_regressor_class.return_value = mock_regressor
        
        mock_predictions = MagicMock()
        mock_regressor.predict.return_value = mock_predictions
        
        # Mock the statistics and predictions
        mock_stats = MagicMock()
        mock_predictions.select_statement.return_value.collect.return_value = mock_stats
        
        mock_predictions.collect.return_value.head.return_value.to_dict.return_value = {
            "id": [1, 2, 3],
            "target_PRED": [10.5, 12.3, 8.7]
        }
        
        # Call the tool in prediction mode
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            model_name="test_model"
        )
        
        # Verify expectations
        mock_regressor_class.assert_called_once()
        mock_regressor.predict.assert_called_once()
        self.assertIn("Made predictions", result)
        self.assertIn("test_model", result)
    
    def test_regression_missing_parameters(self):
        """Test regression tool with missing parameters."""
        # Call without required params
        result = self.tool._run(target_column="target")
        self.assertIn("Error", result)
        
        result = self.tool._run(table_name="TEST_TABLE")
        self.assertIn("Error", result)
    
    @patch("hana_ai.tools.hana_ml_tools.unsupported_tools.UnifiedRegression")
    def test_regression_error_handling(self, mock_regressor_class):
        """Test regression tool error handling."""
        # Setup mock to raise exception
        mock_regressor_class.side_effect = ValueError("Test error")
        
        # Call the tool
        result = self.tool._run(
            table_name="TEST_TABLE",
            target_column="target",
            mode="train"
        )
        
        # Verify error is handled
        self.assertIn("Error", result)
        self.assertIn("Test error", result)


class TestToolInputSchemas:
    """Tests for the input schemas."""
    
    def test_classification_tool_input_validation(self):
        """Test ClassificationToolInput validation."""
        # Valid input
        valid_input = ClassificationToolInput(
            table_name="TEST_TABLE",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            mode="train",
            algorithm="RandomForest"
        )
        assert valid_input.table_name == "TEST_TABLE"
        assert valid_input.mode == "train"
        
        # Invalid mode
        with pytest.raises(ValueError, match="Mode must be either"):
            ClassificationToolInput(
                table_name="TEST_TABLE",
                target_column="target",
                mode="invalid"
            )
        
        # Invalid algorithm
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            ClassificationToolInput(
                table_name="TEST_TABLE",
                target_column="target",
                algorithm="InvalidAlgorithm"
            )
    
    def test_regression_tool_input_validation(self):
        """Test RegressionToolInput validation."""
        # Valid input
        valid_input = RegressionToolInput(
            table_name="TEST_TABLE",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            mode="predict",
            algorithm="GradientBoosting"
        )
        assert valid_input.table_name == "TEST_TABLE"
        assert valid_input.mode == "predict"
        
        # Invalid mode
        with pytest.raises(ValueError, match="Mode must be either"):
            RegressionToolInput(
                table_name="TEST_TABLE",
                target_column="target",
                mode="invalid"
            )
        
        # Invalid algorithm
        with pytest.raises(ValueError, match="Algorithm must be one of"):
            RegressionToolInput(
                table_name="TEST_TABLE",
                target_column="target",
                algorithm="InvalidAlgorithm"
            )