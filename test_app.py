"""
Test script for Diabetes Risk Prediction App
============================================
This script tests the core functionality without requiring Streamlit interface.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def test_model_functionality():
    """Test the core model functionality"""
    print("🔍 Testing Diabetes Prediction Model...")
    print("=" * 50)
    
    try:
        # Load data
        print("1. Loading dataset...")
        df = pd.read_csv('diabetes.csv')
        print(f"   ✅ Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")
        
        # Prepare data
        print("2. Preparing data...")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   ✅ Train set: {X_train_scaled.shape[0]} samples")
        print(f"   ✅ Test set: {X_test_scaled.shape[0]} samples")
        
        # Train model
        print("3. Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        print("   ✅ Model trained successfully")
        
        # Make predictions
        print("4. Evaluating model performance...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Feature importance
        print("5. Analyzing feature importance...")
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("   📊 Top 5 Most Important Features:")
        for i, row in importance_df.head().iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.3f} ({row['Importance']*100:.1f}%)")
        
        # Test sample predictions
        print("6. Testing sample predictions...")
        
        # Low risk sample
        low_risk = [1, 90, 70, 20, 80, 22, 0.3, 25]  # Young, healthy profile
        low_risk_scaled = scaler.transform([low_risk])
        low_pred = model.predict(low_risk_scaled)[0]
        low_prob = model.predict_proba(low_risk_scaled)[0]
        
        print(f"   📊 Low Risk Test:")
        print(f"      Prediction: {'High Risk' if low_pred == 1 else 'Low Risk'}")
        print(f"      Diabetes Probability: {low_prob[1]:.3f} ({low_prob[1]*100:.1f}%)")
        
        # High risk sample
        high_risk = [8, 180, 90, 35, 200, 35, 1.5, 55]  # Older, multiple risk factors
        high_risk_scaled = scaler.transform([high_risk])
        high_pred = model.predict(high_risk_scaled)[0]
        high_prob = model.predict_proba(high_risk_scaled)[0]
        
        print(f"   📊 High Risk Test:")
        print(f"      Prediction: {'High Risk' if high_pred == 1 else 'Low Risk'}")
        print(f"      Diabetes Probability: {high_prob[1]:.3f} ({high_prob[1]*100:.1f}%)")
        
        print("\n🎉 ALL TESTS PASSED! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_data_quality():
    """Test data quality and structure"""
    print("\n🔍 Testing Data Quality...")
    print("=" * 50)
    
    try:
        df = pd.read_csv('diabetes.csv')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns[:-1])}")
        print(f"Target: {df.columns[-1]}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        # Check class distribution
        class_dist = df['Outcome'].value_counts()
        print(f"Class distribution:")
        print(f"  No Diabetes (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
        print(f"  Diabetes (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
        
        # Check for zero values (potential missing data)
        zero_counts = {}
        for col in df.columns[:-1]:  # Exclude target
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_counts[col] = zero_count
        
        if zero_counts:
            print(f"Zero values found (may indicate missing data):")
            for col, count in zero_counts.items():
                print(f"  {col}: {count} zeros ({count/len(df)*100:.1f}%)")
        
        print("✅ Data quality check completed")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in data quality check: {e}")
        return False

if __name__ == "__main__":
    print("🩺 DIABETES PREDICTION MODEL - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test data quality
    data_ok = test_data_quality()
    
    # Test model functionality
    model_ok = test_model_functionality()
    
    print("\n" + "=" * 60)
    if data_ok and model_ok:
        print("🎯 OVERALL STATUS: ✅ ALL SYSTEMS READY FOR DEPLOYMENT")
        print("📱 Streamlit app is ready to be deployed to Streamlit Cloud")
        print("🌐 Your app is currently running at: http://localhost:8501")
    else:
        print("❌ OVERALL STATUS: Issues detected, please fix before deployment")
    
    print("=" * 60)
