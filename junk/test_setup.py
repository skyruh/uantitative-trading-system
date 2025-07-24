#!/usr/bin/env python3
"""
Test script to verify the project structure and configuration setup.
"""

import sys
import os

# Add src to Python path
sys.path.append('src')

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from src.config.settings import config
        from src.utils.logging_utils import get_logger, log_system_startup
        from src.utils.validation_utils import validate_system_dependencies
        from src.interfaces.data_interfaces import IDataSource, IDataStorage
        from src.interfaces.model_interfaces import ILSTMModel, IDQNAgent
        from src.interfaces.trading_interfaces import TradingSignal, Position
        
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_configuration():
    """Test configuration loading and validation."""
    try:
        from src.config.settings import config
        
        print(f"✓ Configuration loaded for environment: {config.environment}")
        print(f"✓ Stock symbols loaded: {len(config.get_stock_symbols())} symbols")
        
        if config.validate_config():
            print("✓ Configuration validation passed")
            return True
        else:
            print("✗ Configuration validation failed")
            return False
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_logging():
    """Test logging setup."""
    try:
        from src.utils.logging_utils import get_logger, log_system_startup
        
        logger = get_logger("TestSetup")
        logger.info("Test log message")
        log_system_startup()
        
        print("✓ Logging system working")
        return True
    except Exception as e:
        print(f"✗ Logging error: {e}")
        return False


def test_validation():
    """Test validation utilities."""
    try:
        from src.utils.validation_utils import validate_system_dependencies
        
        is_valid, issues = validate_system_dependencies()
        if is_valid:
            print("✓ System dependencies validation passed")
            return True
        else:
            print(f"⚠ System dependencies validation found issues: {issues}")
            print("  Note: Install packages with: pip install -r requirements.txt")
            return False
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = ['data', 'models', 'logs', 'config', 'src']
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            all_exist = False
    
    return all_exist


def main():
    """Run all setup tests."""
    print("=" * 50)
    print("Quantitative Trading System - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Logging", test_logging),
        ("Validation", test_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System setup is complete.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)