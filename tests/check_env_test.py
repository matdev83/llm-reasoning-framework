import sys
import os

def test_check_environment_details():
    print(f"Pytest sys.executable: {sys.executable}")
    print(f"Pytest sys.path: {sys.path}")
    print(f"Pytest os.getcwd(): {os.getcwd()}")

    try:
        import requests
        print("Requests module imported successfully by pytest.")
    except ImportError as e:
        print(f"Error importing requests in pytest: {e}")

    assert True # Just to make it a valid test that runs
