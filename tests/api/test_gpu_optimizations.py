"""
Unit tests for GPU optimization features.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import os
import torch
import logging

from hana_ai.api.gpu_utils_hopper import (
    HopperOptimizer, 
    detect_and_optimize_for_hopper
)


class TestHopperDetection(unittest.TestCase):
    """Test Hopper detection functionality."""
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_detect_hopper_compute_capability(self, mock_get_props, mock_device_count, mock_is_available):
        """Test Hopper detection via compute capability."""
        # Set up mock properties with Hopper compute capability (9.0)
        mock_props = MagicMock()
        mock_props.major = 9
        mock_props.minor = 0
        mock_props.name = "NVIDIA H100"
        mock_get_props.return_value = mock_props
        
        # Create optimizer and test detection
        optimizer = HopperOptimizer()
        
        # Verify H100 was detected
        self.assertTrue(optimizer.is_hopper)
        mock_get_props.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_detect_non_hopper(self, mock_get_props, mock_device_count, mock_is_available):
        """Test detection of non-Hopper GPU."""
        # Set up mock properties with A100 compute capability (8.0)
        mock_props = MagicMock()
        mock_props.major = 8
        mock_props.minor = 0
        mock_props.name = "NVIDIA A100"
        mock_get_props.return_value = mock_props
        
        # Create optimizer
        optimizer = HopperOptimizer()
        
        # Verify H100 was not detected
        self.assertFalse(optimizer.is_hopper)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=0)
    def test_detect_no_gpus(self, mock_device_count, mock_is_available):
        """Test detection with no GPUs."""
        # Create optimizer with no GPUs
        optimizer = HopperOptimizer()
        
        # Verify not detected as Hopper
        self.assertFalse(optimizer.is_hopper)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_detect_no_cuda(self, mock_is_available):
        """Test detection with CUDA not available."""
        # Create optimizer with CUDA not available
        optimizer = HopperOptimizer()
        
        # Verify not detected as Hopper
        self.assertFalse(optimizer.is_hopper)


@pytest.mark.parametrize("is_hopper,expected_keys", [
    (True, ['use_fp8', 'transformer_engine', 'enable_gptq', 'enable_awq']),
    (False, {})
])
def test_get_hopper_specific_args(is_hopper, expected_keys):
    """Test hopper-specific args are returned properly."""
    with patch.object(HopperOptimizer, '_detect_hopper', return_value=is_hopper):
        optimizer = HopperOptimizer()
        args = optimizer.get_hopper_specific_args()
        
        if is_hopper:
            # Check that all expected keys are present
            for key in expected_keys:
                assert key in args
        else:
            # Empty dict for non-Hopper
            assert args == {}


class TestQuantization(unittest.TestCase):
    """Test quantization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a patch for Hopper detection to always return True
        self.hopper_patcher = patch.object(HopperOptimizer, '_detect_hopper', return_value=True)
        self.mock_detect_hopper = self.hopper_patcher.start()
        
        # Initialize optimizer with quantization settings
        self.optimizer = HopperOptimizer(
            enable_gptq=True,
            enable_awq=True,
            quantization_bit_width=4,
            quantization_cache_dir="/tmp/test_quantization"
        )
        
        # Create directory for testing
        os.makedirs("/tmp/test_quantization", exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.hopper_patcher.stop()
    
    @patch('hana_ai.api.gpu_utils_hopper.GPTQ_AVAILABLE', True)
    @patch('hana_ai.api.gpu_utils_hopper.AutoGPTQForCausalLM')
    def test_gptq_quantization(self, mock_gptq_class):
        """Test GPTQ quantization."""
        # Setup mock
        mock_gptq = MagicMock()
        mock_gptq_class.from_pretrained.return_value = mock_gptq
        
        # Call quantize method
        self.optimizer.quantize_with_gptq(
            model="test_model",
            model_name="test_model"
        )
        
        # Verify GPTQ was called correctly
        mock_gptq_class.from_pretrained.assert_called_once()
        mock_gptq.quantize.assert_called_once()
    
    @patch('hana_ai.api.gpu_utils_hopper.AWQ_AVAILABLE', True)
    @patch('hana_ai.api.gpu_utils_hopper.AutoAWQForCausalLM')
    def test_awq_quantization(self, mock_awq_class):
        """Test AWQ quantization."""
        # Setup mock
        mock_awq = MagicMock()
        mock_awq_class.from_pretrained.return_value = mock_awq
        
        # Call quantize method
        self.optimizer.quantize_with_awq(
            model="test_model",
            model_name="test_model"
        )
        
        # Verify AWQ was called correctly
        mock_awq_class.from_pretrained.assert_called_once()
        mock_awq.quantize.assert_called_once()
    
    def test_calibration_data_generation(self):
        """Test calibration data generation."""
        # Get default calibration data
        data = self.optimizer._get_default_calibration_data(num_samples=10)
        
        # Verify the correct number of samples
        self.assertEqual(len(data), 10)
        
        # Verify the samples are strings
        for sample in data:
            self.assertIsInstance(sample, str)
    
    @patch('hana_ai.api.gpu_utils_hopper.random.random', return_value=0.1)  # Below 0.3 threshold
    def test_domain_specific_calibration_no_variation(self, mock_random):
        """Test domain-specific calibration without variations."""
        # Get finance domain calibration data
        with patch.object(HopperOptimizer, '_detect_domain', return_value="finance"):
            data = self.optimizer._get_default_calibration_data(num_samples=5)
            
            # Get the base samples for comparison
            finance_samples = self.optimizer._get_finance_calibration_data()
            
            # Verify data contains exact samples from finance domain without variation
            for i in range(min(5, len(finance_samples))):
                self.assertEqual(data[i], finance_samples[i])
    
    @patch('hana_ai.api.gpu_utils_hopper.random.random', return_value=0.5)  # Above 0.3 threshold
    @patch('hana_ai.api.gpu_utils_hopper.random.choice')
    def test_domain_specific_calibration_with_variation(self, mock_choice, mock_random):
        """Test domain-specific calibration with variations."""
        # Mock the random choice to return a specific variation function
        def mock_variation(s):
            return f"Modified: {s}"
        
        mock_choice.return_value = mock_variation
        
        # Get analytics domain calibration data
        with patch.object(HopperOptimizer, '_detect_domain', return_value="analytics"):
            data = self.optimizer._get_default_calibration_data(num_samples=5)
            
            # Verify data contains variations
            for item in data:
                self.assertTrue(item.startswith("Modified:"))


class TestTransformerEngine(unittest.TestCase):
    """Test Transformer Engine initialization."""
    
    @patch('hana_ai.api.gpu_utils_hopper.TE_AVAILABLE', True)
    @patch('hana_ai.api.gpu_utils_hopper.te')
    def test_transformer_engine_initialization(self, mock_te):
        """Test Transformer Engine initialization."""
        # Setup mock properties
        mock_te.common.recipe.Format.E4M3 = "E4M3"
        mock_te.common.recipe.Format.E5M2 = "E5M2"
        mock_te.common.recipe.DelayedScaling = MagicMock(return_value="recipe")
        
        # Create mock implementation of dir
        def mock_dir(obj):
            return ["common", "pytorch", "register_fp8_operators"]
        
        # Create optimizer with mocked detection
        with patch.object(HopperOptimizer, '_detect_hopper', return_value=True):
            optimizer = HopperOptimizer(enable_fp8=True)
            
            # Mock dir function
            with patch('hana_ai.api.gpu_utils_hopper.dir', mock_dir):
                # Initialize Transformer Engine
                result = optimizer._init_transformer_engine()
                
                # Verify Transformer Engine was initialized
                self.assertTrue(result)
                mock_te.common.recipe.DelayedScaling.assert_called_once()
                self.assertEqual(optimizer.fp8_recipe, "recipe")
    
    @patch('hana_ai.api.gpu_utils_hopper.TE_AVAILABLE', False)
    def test_transformer_engine_not_available(self):
        """Test when Transformer Engine is not available."""
        # Create optimizer with TE not available
        with patch.object(HopperOptimizer, '_detect_hopper', return_value=True):
            optimizer = HopperOptimizer(enable_fp8=True)
            
            # Initialize Transformer Engine
            result = optimizer._init_transformer_engine()
            
            # Verify initialization failed
            self.assertFalse(result)


class TestOptimizerIntegration(unittest.TestCase):
    """Integration tests for the HopperOptimizer."""
    
    @patch('hana_ai.api.gpu_utils_hopper.HopperOptimizer._detect_hopper')
    def test_detect_and_optimize_for_hopper(self, mock_detect_hopper):
        """Test the high-level detect_and_optimize_for_hopper function."""
        # Test when Hopper is detected
        mock_detect_hopper.return_value = True
        is_hopper, args = detect_and_optimize_for_hopper()
        
        # Verify Hopper was detected and args returned
        self.assertTrue(is_hopper)
        self.assertIsInstance(args, dict)
        self.assertIn("use_fp8", args)
        
        # Test when Hopper is not detected
        mock_detect_hopper.return_value = False
        is_hopper, args = detect_and_optimize_for_hopper()
        
        # Verify Hopper was not detected
        self.assertFalse(is_hopper)
        self.assertEqual(args, {})