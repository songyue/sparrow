#!/usr/bin/env python3
"""
Simple test to verify logging statements are present in the code.
This test doesn't require dependencies to be installed.
"""

import re
import sys
from pathlib import Path

def check_logging_in_file(filepath, required_patterns):
    """Check if required logging patterns are present in file."""
    print(f"\nChecking {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    found_patterns = []
    missing_patterns = []
    
    for pattern_name, pattern in required_patterns.items():
        if re.search(pattern, content, re.MULTILINE):
            found_patterns.append(pattern_name)
            print(f"  ✓ Found: {pattern_name}")
        else:
            missing_patterns.append(pattern_name)
            print(f"  ✗ Missing: {pattern_name}")
    
    return len(missing_patterns) == 0, found_patterns, missing_patterns

def test_huggingface_logging():
    """Test that HuggingFace inference has comprehensive logging."""
    print("=" * 80)
    print("Testing HuggingFace Inference Logging")
    print("=" * 80)
    
    filepath = Path(__file__).parent / "sparrow_parse" / "vllm" / "huggingface_inference.py"
    
    required_patterns = {
        "logging import": r"import logging",
        "datetime import": r"from datetime import datetime",
        "logger setup": r"self\.logger\s*=\s*logging\.getLogger",
        "logger level": r"self\.logger\.setLevel\(logging\.INFO\)",
        "starting inference log": r'self\.logger\.info.*Starting HuggingFace inference',
        "connection log": r'self\.logger\.info.*Connecting to HuggingFace Space',
        "connection success log": r'self\.logger\.info.*Successfully connected',
        "connection time": r'connection_time.*total_seconds\(\)',
        "file processing log": r'self\.logger\.info.*Processing.*file',
        "request log": r'self\.logger\.info.*Sending prediction request',
        "response time": r'request_time.*total_seconds\(\)',
        "error handling": r'self\.logger\.error.*Failed to',
        "error type logging": r'self\.logger\.error.*Error type',
    }
    
    success, found, missing = check_logging_in_file(filepath, required_patterns)
    
    print(f"\nResult: {len(found)}/{len(required_patterns)} patterns found")
    
    return success

def test_ollama_logging():
    """Test that Ollama inference has comprehensive logging."""
    print("\n" + "=" * 80)
    print("Testing Ollama Inference Logging")
    print("=" * 80)
    
    filepath = Path(__file__).parent / "sparrow_parse" / "vllm" / "ollama_inference.py"
    
    required_patterns = {
        "logging import": r"import logging",
        "datetime import": r"from datetime import datetime",
        "logger setup": r"self\.logger\s*=\s*logging\.getLogger",
        "logger level": r"self\.logger\.setLevel\(logging\.INFO\)",
        "starting text inference log": r'self\.logger\.info.*Starting Ollama text inference',
        "starting image inference log": r'self\.logger\.info.*Starting Ollama image inference',
        "model log": r'self\.logger\.info.*Model:',
        "request log": r'self\.logger\.info.*Sending request to Ollama',
        "response time": r'request_time.*total_seconds\(\)',
        "processing image log": r'self\.logger\.info.*Processing image',
        "error handling": r'self\.logger\.error.*Error',
        "error type logging": r'self\.logger\.error.*Error type',
    }
    
    success, found, missing = check_logging_in_file(filepath, required_patterns)
    
    print(f"\nResult: {len(found)}/{len(required_patterns)} patterns found")
    
    return success

def main():
    """Run all tests."""
    print("\nNetwork Request Logging Code Verification")
    print("=" * 80)
    
    tests = [
        test_huggingface_logging,
        test_ollama_logging,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✓ All tests passed ({passed}/{total})")
        print("\nLogging has been successfully added to:")
        print("  - HuggingFace inference (network requests via gradio_client)")
        print("  - Ollama inference (network requests via ollama API)")
        print("\nThe logging includes:")
        print("  - Connection attempts and timing")
        print("  - File processing information")
        print("  - Request/response timing")
        print("  - Error handling with detailed error types")
        return True
    else:
        print(f"✗ Some tests failed ({passed}/{total} passed)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
