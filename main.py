# main.py
from src.evaluator import run_demo

if __name__ == "__main__":
    print("🎯 Enron Email Multi-Label Triage System")
    print("=========================================\n")
    
    run_demo()
    
    print("\n✅ System ready! You can now:")
    print("   - Use the trained models for predictions")
    print("   - Extend with real Enron data")
    print("   - Add more categories and features")