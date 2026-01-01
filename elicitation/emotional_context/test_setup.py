#!/usr/bin/env python3
"""
Quick setup test to validate the environment and API access.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import config
        import schemas
        import utils
        import judge_prompts
        print("✓ All local imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_api_key():
    """Test that API key is configured."""
    print("\nTesting API configuration...")
    import config
    if config.ANTHROPIC_API_KEY:
        print("✓ ANTHROPIC_API_KEY is set")
        return True
    else:
        print("✗ ANTHROPIC_API_KEY is not set")
        print("  Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    import config
    dirs = [config.DATA_DIR, config.LOGS_DIR, config.OUTPUTS_DIR]
    all_exist = True
    for dir_path in dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.name}/ exists")
        else:
            print(f"✗ {dir_path.name}/ does not exist")
            all_exist = False
    return all_exist


def test_model_constants():
    """Test that model constants are defined."""
    print("\nTesting model constants...")
    try:
        from utils import opus_4_5, sonnet_4_5
        print("✓ Model constants defined")
        print(f"  opus_4_5: {opus_4_5}")
        print(f"  sonnet_4_5: {sonnet_4_5}")
        return True
    except Exception as e:
        print(f"✗ Model constants failed: {e}")
        return False


def test_anthropic_client():
    """Test that Anthropic client can be instantiated."""
    print("\nTesting Anthropic client...")
    try:
        import anthropic
        import config
        if config.ANTHROPIC_API_KEY:
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            print("✓ Anthropic client instantiated successfully")
            return True
        else:
            print("✗ Cannot test client without API key")
            return False
    except Exception as e:
        print(f"✗ Client instantiation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Emotional Context Evaluation System - Setup Test")
    print("=" * 60)

    results = [
        test_imports(),
        test_api_key(),
        test_directories(),
        test_model_constants(),
        test_anthropic_client()
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed! System is ready to run.")
        print("\nTo run the pipeline:")
        print("  python run_all_stages.py")
        print("\nOr run individual stages:")
        print("  python stage1_generate_neutral_requests.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
