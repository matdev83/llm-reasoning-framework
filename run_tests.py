#!/usr/bin/env python3
"""
Test runner script for the LLM Reasoning Framework project.

This script provides convenient ways to run different types of tests:
- Unit tests (default)
- Integration tests (requires API keys)
- All tests
- Specific test modules or functions

Usage:
    python run_tests.py                    # Run unit tests only
    python run_tests.py --integration      # Run integration tests only  
    python run_tests.py --all              # Run all tests
    python run_tests.py --hybrid           # Run hybrid-related tests only
    python run_tests.py --far              # Run FaR-related tests only
    python run_tests.py --far --integration # Run FaR tests including real integration tests
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --coverage         # Run with coverage report
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
    else:
        print(f"❌ {description} - FAILED")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run tests for LLM Reasoning Framework project")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests (requires OPENROUTER_API_KEY)")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests including integration tests")
    parser.add_argument("--hybrid", action="store_true",
                       help="Run hybrid-related tests only")
    parser.add_argument("--far", action="store_true",
                       help="Run FaR-related tests only")
    parser.add_argument("--functionality", action="store_true",
                       help="Run functionality tests that catch method signature issues")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Run with verbose output")
    parser.add_argument("--coverage", action="store_true",
                       help="Run with coverage report")
    parser.add_argument("--module", type=str,
                       help="Run specific test module (e.g., 'tests.hybrid.test_hybrid_processor')")
    parser.add_argument("--function", type=str,
                       help="Run specific test function (e.g., 'test_hybrid_processor_method_signatures')")
    
    args = parser.parse_args()
    
    # Base pytest command - use sys.executable to ensure we use the right Python
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    # Determine what tests to run
    if args.all:
        cmd.append("-m")
        cmd.append("integration or not integration")
        description = "Running ALL tests (unit + integration)"
    elif args.integration:
        cmd.append("-m")
        cmd.append("integration")
        description = "Running INTEGRATION tests only"
        # Check for API key
        if not os.getenv('OPENROUTER_API_KEY'):
            print("⚠️  WARNING: OPENROUTER_API_KEY environment variable not set")
            print("   Integration tests may be skipped or fail")
    elif args.hybrid:
        cmd.append("tests/hybrid/")
        if args.functionality:
            cmd.append("tests/hybrid/test_hybrid_functionality.py")
        description = "Running HYBRID tests only"
    elif args.far:
        cmd.append("tests/far/")
        cmd.append("tests/integration/test_far_integration.py")
        if args.integration:
            cmd.append("tests/integration/test_far_real_integration.py")
        description = "Running FaR tests only"
    elif args.functionality:
        cmd.append("tests/hybrid/test_hybrid_method_signatures.py")
        description = "Running FUNCTIONALITY tests (method signature validation)"
    elif args.module:
        cmd.append(args.module.replace(".", "/") + ".py")
        description = f"Running module: {args.module}"
    elif args.function:
        cmd.append("-k")
        cmd.append(args.function)
        description = f"Running function: {args.function}"
    else:
        # Default: run unit tests only (integration tests excluded by pytest.ini)
        description = "Running UNIT tests only (integration tests excluded)"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        
    # If running integration tests, provide helpful info
    if args.integration or args.all:
        print("\n📋 Integration Test Info:")
        print("   - These tests make real API calls to OpenRouter")
        print("   - They test end-to-end functionality with actual models")
        print("   - Requires OPENROUTER_API_KEY environment variable")
        print("   - Tests use: deepseek/deepseek-r1-0528:free and openrouter/cypher-alpha:free")
        print("\n📋 Available Integration Tests:")
        print("   - Hybrid: tests/integration/test_hybrid_integration.py (2 tests)")
        print("   - FaR: tests/integration/test_far_real_integration.py (9 tests)")
        print("   - FaR (mocked): tests/integration/test_far_integration.py (5 tests)")

if __name__ == "__main__":
    main() 