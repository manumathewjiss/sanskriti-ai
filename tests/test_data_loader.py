"""
Unit tests for data_loader module.
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import (
    load_csv_data,
    load_and_chunk_itihasa_data,
    validate_csv_columns,
    get_unique_books
)


class TestDataLoader:
    """Test cases for data loading functions."""
    
    def test_load_csv_nonexistent_file(self):
        """Test loading non-existent CSV file."""
        df = load_csv_data("nonexistent_file.csv")
        assert df is None
    
    def test_load_and_chunk_empty_list(self):
        """Test loading with no documents."""
        docs = load_and_chunk_itihasa_data("nonexistent.csv")
        assert docs == []
    
    def test_validate_csv_columns(self):
        """Test CSV column validation."""
        import pandas as pd
        
        # Create test dataframe
        df = pd.DataFrame({
            'sn': ['test'],
            'en': ['test'],
            'book': ['Test Book'],
            'chapter': [1],
            'shloka': [1]
        })
        
        required = ['sn', 'en', 'book', 'chapter', 'shloka']
        assert validate_csv_columns(df, required) is True
    
    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        import pandas as pd
        
        df = pd.DataFrame({'sn': ['test']})
        required = ['sn', 'en', 'book']
        
        assert validate_csv_columns(df, required) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
