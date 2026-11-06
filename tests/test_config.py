"""
Unit tests for configuration module.
"""

import pytest
import sys
import os
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config


class TestConfig:
    """Test cases for configuration settings."""
    
    def test_config_exists(self):
        """Test that Config class has required attributes."""
        assert hasattr(Config, 'PROJECT_DIR')
        assert hasattr(Config, 'BASE_MODEL')
        assert hasattr(Config, 'MAX_TOKENS')
        assert hasattr(Config, 'TEMPERATURE')
    
    def test_get_bnb_config(self):
        """Test BitsAndBytes configuration."""
        bnb_config = Config.get_bnb_config()
        assert bnb_config is not None
        assert bnb_config.load_in_4bit is True
    
    def test_cuda_availability(self):
        """Test CUDA availability check."""
        available = Config.is_cuda_available()
        assert isinstance(available, bool)
    
    def test_config_values_reasonable(self):
        """Test that configuration values are reasonable."""
        assert Config.MAX_TOKENS > 0
        assert 0 <= Config.TEMPERATURE <= 1
        assert 0 < Config.TEMPERATURE < 1 or Config.TEMPERATURE == 0
        assert Config.RETRIEVER_K > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
