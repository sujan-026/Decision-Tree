#!/usr/bin/env python3
"""
Test script for the enhanced medical triage system.
Validates training, model loading, and prediction functionality.
"""

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

def test_import_modules():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        import preprocessor
        import train
        import triage_cli_enhanced
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_availability():
    """Test if required data files are available."""
    print("\n📁 Testing data availability...")
    
    data_dir = Path("data")
    required_files = [
        "dataset.csv",
        "Symptom-severity.csv", 
        "symptom_Description.csv",
        "symptom_precaution.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("   Please place these files in the 'data/' directory")
        return False
    else:
        print("✅ All required data files found")
        return True

def test_data_loading():
    """Test data loading and preprocessing functions."""
    print("\n📊 Testing data loading...")
    
    try:
        from preprocessor import load_raw, clean_dataset
        
        data_dir = Path("data")
        df, sev_map, desc_df, prec_df = load_raw(data_dir)
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✅ Severity map: {len(sev_map)} symptoms")
        print(f"✅ Descriptions: {len(desc_df)} diseases")
        print(f"✅ Precautions: {len(prec_df)} diseases")
        
        # Test cleaning
        df_clean = clean_dataset(df)
        print(f"✅ Data cleaned successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test enhanced feature matrix building."""
    print("\n🔧 Testing feature engineering...")
    
    try:
        from preprocessor import load_raw, clean_dataset, build_enhanced_feature_matrix
        
        data_dir = Path("data")
        df, sev_map, desc_df, prec_df = load_raw(data_dir)
        df_clean = clean_dataset(df)
        
        # Take a sample for faster testing
        df_sample = df_clean.sample(min(100, len(df_clean)), random_state=42)
        
        X, y, demo_pipe, mlb, feat_names, feat_weights = build_enhanced_feature_matrix(df_sample, sev_map)
        
        print(f"✅ Feature matrix created: {X.shape}")
        print(f"✅ Features: {len(feat_names)} total")
        print(f"✅ Classes: {len(np.unique(y))} diseases")
        print(f"✅ Feature weights: {len(feat_weights)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        traceback.print_exc()
        return False

def test_model_training():
    """Test if training can run (using small sample)."""
    print("\n🏋️ Testing model training (sample data)...")
    
    try:
        from preprocessor import load_raw, clean_dataset, build_enhanced_feature_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        
        data_dir = Path("data")
        df, sev_map, desc_df, prec_df = load_raw(data_dir)
        df_clean = clean_dataset(df)
        
        # Use small sample for quick test
        df_sample = df_clean.sample(min(200, len(df_clean)), random_state=42)
        
        X, y, demo_pipe, mlb, feat_names, feat_weights = build_enhanced_feature_matrix(df_sample, sev_map)
        
        # Quick training test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training successful")
        print(f"✅ Test accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if existing models can be loaded."""
    print("\n📦 Testing model loading...")
    
    models_dir = Path("models")
    model_files = ["triage_model.joblib", "triage_model_enhanced.joblib"]
    
    loaded_models = 0
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                import joblib
                model_bundle = joblib.load(model_path)
                print(f"✅ {model_file} loaded successfully")
                
                # Check bundle contents
                if "model" in model_bundle or "tree" in model_bundle:
                    print(f"   - Model object found")
                if "mlb" in model_bundle:
                    print(f"   - MultiLabelBinarizer found")
                if "feature_names" in model_bundle:
                    print(f"   - {len(model_bundle['feature_names'])} features")
                
                loaded_models += 1
                
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
        else:
            print(f"⚠️ {model_file} not found (run training first)")
    
    return loaded_models > 0

def test_cli_functions():
    """Test CLI helper functions."""
    print("\n🖥️ Testing CLI functions...")
    
    try:
        from triage_cli_enhanced import (
            ask_symptom_with_context,
            get_contextual_questions, 
            calculate_risk_score,
            get_demographic_risk_factors
        )
        from preprocessor import get_symptom_severity_score
        
        # Test demographic risk factors
        risk_factors = get_demographic_risk_factors(70, 32, "female")
        print(f"✅ Risk factors calculation: {risk_factors}")
        
        # Test severity score
        severity_map = {"fever": 4, "headache": 3, "cough": 2}
        symptoms = ["fever", "headache"]
        score = get_symptom_severity_score(symptoms, severity_map)
        print(f"✅ Severity score calculation: {score}")
        
        # Test contextual questions
        current_symptoms = ["fever", "headache"]
        questions = get_contextual_questions(current_symptoms)
        print(f"✅ Contextual questions: {len(questions)} generated")
        
        # Test risk calculation
        demographics = {"Age": 65, "Height_cm": 170, "Weight_kg": 80, "Gender": "male"}
        risk_assessment = calculate_risk_score(demographics, symptoms)
        print(f"✅ Risk assessment: {risk_assessment['level']} ({risk_assessment['score']:.1f})")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI function error: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🚀 Enhanced Medical Triage System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_import_modules),
        ("Data Availability", test_data_availability),
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Model Loading", test_model_loading),
        ("CLI Functions", test_cli_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your enhanced system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python train.py' to train the enhanced model")
        print("2. Run 'python triage_cli_enhanced.py' to use the enhanced CLI")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        
        if not results[1][1]:  # Data availability failed
            print("\n📁 Missing data files. Please ensure you have:")
            print("   - data/dataset.csv")
            print("   - data/Symptom-severity.csv")
            print("   - data/symptom_Description.csv") 
            print("   - data/symptom_precaution.csv")

def main():
    """Main test execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test - just imports and data
        print("🏃 Quick Test Mode")
        test_import_modules()
        test_data_availability()
    else:
        # Comprehensive test
        run_comprehensive_test()

if __name__ == "__main__":
    main()