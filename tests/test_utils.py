"""
Unit tests for utility functions.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    truncate_text,
    clean_text,
    validate_query,
    chunk_text,
    devanagari_to_roman
)


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a very long text that needs truncation"
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncation of short text."""
        text = "Short"
        truncated = truncate_text(text, max_length=20)
        assert truncated == text
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Extra   spaces   and  \n\n  newlines  "
        cleaned = clean_text(text)
        assert cleaned == "Extra spaces and newlines"
    
    def test_validate_query_valid(self):
        """Test query validation with valid input."""
        assert validate_query("What is dharma?") is True
    
    def test_validate_query_empty(self):
        """Test query validation with empty input."""
        with pytest.raises(ValueError):
            validate_query("")
    
    def test_validate_query_too_long(self):
        """Test query validation with too long input."""
        long_query = "a" * 2000
        with pytest.raises(ValueError):
            validate_query(long_query, max_length=1000)
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
