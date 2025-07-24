#!/usr/bin/env python3
"""
Integration test runner for the Quantitative Trading System.
Executes all integration tests and validates system functionality.
"""

import os
import sys
import unittest
import argparse
from datetime import datetime

# Add src to Python path
sys.path.append('src')

from src.utils.logging_utils import get_logger


def run_integration_tests(verbose=False):
    """Run all integration tests."""
    logger = get_logger("TestRunner")
    
    logger.info("=" * 60)
    logger.info("Starting Integration Tests")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add integration tests
    integration_tests = test_loader.discover('tests', pattern='test_integration.py')
    system_validation_tests = test_loader.discover('tests', pattern='test_system_validation.py')
    
    test_suite.addTests(integration_tests)
    test_suite.addTests(system_validation_tests)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = test_runner.run(test_suite)
    
    # Log results
    logger.info("=" * 60)
    logger.info(f"Test Results: Ran {result.testsRun} tests")
    logger.info(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info("=" * 60)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integration Test Runner")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = run_integration_tests(verbose=args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())