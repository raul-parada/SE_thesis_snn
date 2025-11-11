if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CI/CD AUTOMATED TEST SUITE")
    print("=" * 70 + "\n")
    
    # Simplest possible pytest call
    exit_code = pytest.main(['ci_cd_test.py'])
    
    report = generate_report(exit_code)
    
    print("\n" + "=" * 70)
    print("RESULT:", "PASSED" if exit_code == 0 else "FAILED")
    print("=" * 70 + "\n")
    
    sys.exit(exit_code)
