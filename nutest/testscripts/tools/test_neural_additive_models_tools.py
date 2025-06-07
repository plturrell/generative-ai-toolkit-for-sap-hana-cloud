"""
Test for Neural Additive Models tools.
"""
import os
import pytest
from hana_ai.tools.hana_ml_tools.neural_additive_models_tools import (
    NeuralAdditiveModelsFitAndSave,
    NeuralAdditiveModelsLoadModelAndPredict,
    NeuralAdditiveModelsFeatureImportance
)

from .testML_BaseTestClass import BaseMLTest


class TestNeuralAdditiveModelsTools(BaseMLTest):
    """
    Test class for Neural Additive Models tools.
    """
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        super().setup_class()
        
        # Set test tables and model names - these must exist in your test environment
        # pointing to actual tables with appropriate data
        cls.train_table = os.environ.get("NAM_TEST_TRAIN_TABLE", "SALES_FORECAST_TRAIN")
        cls.test_table = os.environ.get("NAM_TEST_TEST_TABLE", "SALES_FORECAST_TEST")
        cls.model_name = "nam_test_model"
        cls.target_column = os.environ.get("NAM_TEST_TARGET", "SALES")
        
        # Initialize tools
        cls.fit_tool = NeuralAdditiveModelsFitAndSave(connection_context=cls.cc)
        cls.predict_tool = NeuralAdditiveModelsLoadModelAndPredict(connection_context=cls.cc)
        cls.importance_tool = NeuralAdditiveModelsFeatureImportance(connection_context=cls.cc)

    @classmethod
    def teardown_class(cls):
        """Tear down test class."""
        try:
            # Drop any prediction results tables
            for table in [f"{cls.model_name}_1_NAM_PREDICTIONS"]:
                if cls.cc.has_table(table):
                    cls.cc.drop_table(table)
        except Exception as e:
            print(f"Error in teardown: {str(e)}")
        
        super().teardown_class()

    @pytest.mark.skipif("SKIP_NEURAL_ADDITIVE_MODELS" in os.environ, 
                      reason="Skip NAM tests when PyTorch is not available")
    def test_nam_fit_and_save(self):
        """Test fitting and saving a Neural Additive Model."""
        # Skip if the required tables don't exist
        if not self.cc.has_table(self.train_table):
            pytest.skip(f"Test table {self.train_table} not found")
            
        # Test if PyTorch is available, skip if not
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            
        if not has_torch:
            pytest.skip("PyTorch not available")
        
        # Run fit and save
        result = self.fit_tool._run(
            fit_table=self.train_table,
            name=self.model_name,
            target=self.target_column,
            hidden_sizes="32,16",
            num_basis_functions=32,
            epochs=20,
            batch_size=32
        )
        
        # Check result
        import json
        result_dict = json.loads(result)
        assert "trained_table" in result_dict
        assert "model_storage_name" in result_dict
        assert "model_storage_version" in result_dict
        assert result_dict["model_storage_name"] == self.model_name
        
    @pytest.mark.skipif("SKIP_NEURAL_ADDITIVE_MODELS" in os.environ, 
                      reason="Skip NAM tests when PyTorch is not available")
    def test_nam_predict(self):
        """Test predictions with a Neural Additive Model."""
        # Skip if the required tables don't exist
        if not self.cc.has_table(self.test_table):
            pytest.skip(f"Test table {self.test_table} not found")
            
        # Test if PyTorch is available, skip if not
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            
        if not has_torch:
            pytest.skip("PyTorch not available")
            
        # First ensure model exists by running fit if needed
        if not hasattr(self, "model_fitted") or not self.model_fitted:
            self.test_nam_fit_and_save()
            self.model_fitted = True
        
        # Run prediction
        result = self.predict_tool._run(
            predict_table=self.test_table,
            name=self.model_name,
            include_contributions=True
        )
        
        # Check result
        import json
        result_dict = json.loads(result)
        assert "predicted_results_table" in result_dict
        assert result_dict["contributions_included"] is True
        
        # Check if predictions table exists
        pred_table = result_dict["predicted_results_table"]
        assert self.cc.has_table(pred_table)
        
    @pytest.mark.skipif("SKIP_NEURAL_ADDITIVE_MODELS" in os.environ, 
                      reason="Skip NAM tests when PyTorch is not available")
    def test_feature_importance(self):
        """Test getting feature importance from a Neural Additive Model."""
        # Test if PyTorch is available, skip if not
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            
        if not has_torch:
            pytest.skip("PyTorch not available")
            
        # First ensure model exists by running fit if needed
        if not hasattr(self, "model_fitted") or not self.model_fitted:
            self.test_nam_fit_and_save()
            self.model_fitted = True
        
        # Get feature importance
        result = self.importance_tool._run(
            name=self.model_name,
            top_n=3
        )
        
        # Check result
        import json
        result_dict = json.loads(result)
        assert "feature_importance" in result_dict
        assert len(result_dict["feature_importance"]) <= 3
        
        # Check structure of feature importance
        for feat in result_dict["feature_importance"]:
            assert "feature" in feat
            assert "importance" in feat
            assert isinstance(feat["importance"], float)