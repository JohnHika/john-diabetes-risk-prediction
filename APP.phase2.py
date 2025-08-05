"""
Diabetes Risk Prediction - Phase 2: Data Preprocessing and Visualization
========================================================================

PROJECT: Diabetes Risk Prediction through Supervised Machine Learning
PHASE: 2 - Data Preprocessing and Visualization
AUTHOR: John Hika
DATE: August 5, 2025

PHASE 2 OBJECTIVES:
1. Configure your data visualization tool (optional)
2. Visualize feature/feature relationships and draw your conclusion
3. Visualize feature/target relationships and draw your conclusion
4. Select an appropriate model and tell us why
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure visualization tools
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("🩺 DIABETES RISK PREDICTION - PHASE 2")
print("=" * 60)
print("📊 PHASE 2: DATA PREPROCESSING AND VISUALIZATION")
print("=" * 60)

# ============================================================================
# 1. CONFIGURE DATA VISUALIZATION TOOL
# ============================================================================

print("\n1️⃣ DATA VISUALIZATION CONFIGURATION")
print("-" * 50)

print("✅ Visualization Tools Configured:")
print("   • Matplotlib: Static plots with custom styling")
print("   • Seaborn: Statistical visualizations")
print("   • Plotly: Interactive visualizations")
print("   • Figure size: 12x8 inches")
print("   • Color palette: Husl (vibrant colors)")
print("   • Font size: 12pt for readability")

# Load the dataset
try:
    df = pd.read_csv('diabetes.csv')
    print(f"\n📊 Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("\n❌ Error: diabetes.csv not found!")
    print("Please ensure the dataset is in the current directory.")
    exit()

# ============================================================================
# 2. VISUALIZE FEATURE/FEATURE RELATIONSHIPS
# ============================================================================

print("\n2️⃣ FEATURE/FEATURE RELATIONSHIPS ANALYSIS")
print("-" * 50)

# Correlation Matrix
correlation_matrix = df.corr()
print("\n📈 Feature Correlation Matrix:")
print(correlation_matrix.round(3))

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.3f',
            square=True,
            cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix\n(Lower Triangle)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Identify strong correlations
print("\n🔍 FEATURE/FEATURE RELATIONSHIP ANALYSIS:")
print("-" * 40)

strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        correlation = correlation_matrix.iloc[i, j]
        if abs(correlation) > 0.3:  # Strong correlation threshold
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            strong_correlations.append((feature1, feature2, correlation))

print("Strong Feature Correlations (|r| > 0.3):")
if strong_correlations:
    for feat1, feat2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
        strength = "Very Strong" if abs(corr) > 0.7 else "Strong" if abs(corr) > 0.5 else "Moderate"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"   • {feat1} ↔ {feat2}: r = {corr:.3f} ({strength} {direction})")
else:
    print("   • No strong correlations found between features")

# Pairplot for key relationships
print("\n📊 Creating pairplot for feature relationships...")
# Select subset of features for clarity
key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Outcome']
plt.figure(figsize=(15, 12))
sns.pairplot(df[key_features], hue='Outcome', diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairwise Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# ============================================================================
# 3. VISUALIZE FEATURE/TARGET RELATIONSHIPS
# ============================================================================

print("\n3️⃣ FEATURE/TARGET RELATIONSHIPS ANALYSIS")
print("-" * 50)

# Target variable distribution
outcome_counts = df['Outcome'].value_counts()
print(f"\n🎯 Target Variable Distribution:")
print(f"   • No Diabetes (0): {outcome_counts[0]} samples ({outcome_counts[0]/len(df)*100:.1f}%)")
print(f"   • Diabetes (1): {outcome_counts[1]} samples ({outcome_counts[1]/len(df)*100:.1f}%)")

# Plot target distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Target distribution pie chart
axes[0,0].pie(outcome_counts.values, labels=['No Diabetes', 'Diabetes'], 
              autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
axes[0,0].set_title('Target Variable Distribution', fontweight='bold')

# Target distribution bar chart
axes[0,1].bar(['No Diabetes', 'Diabetes'], outcome_counts.values, 
              color=['lightgreen', 'lightcoral'])
axes[0,1].set_title('Target Variable Counts', fontweight='bold')
axes[0,1].set_ylabel('Count')

# Feature correlations with target
target_correlations = correlation_matrix['Outcome'].drop('Outcome').sort_values(key=abs, ascending=False)
axes[1,0].barh(target_correlations.index, target_correlations.values, 
               color=['red' if x < 0 else 'blue' for x in target_correlations.values])
axes[1,0].set_title('Feature Correlations with Diabetes', fontweight='bold')
axes[1,0].set_xlabel('Correlation Coefficient')

# Box plot for top correlated feature
top_feature = target_correlations.index[0]
sns.boxplot(data=df, x='Outcome', y=top_feature, ax=axes[1,1])
axes[1,1].set_title(f'{top_feature} Distribution by Diabetes Status', fontweight='bold')
axes[1,1].set_xticklabels(['No Diabetes', 'Diabetes'])

plt.tight_layout()
plt.show()

print(f"\n🔍 FEATURE/TARGET RELATIONSHIP ANALYSIS:")
print("-" * 40)
print("Feature correlations with Diabetes (sorted by strength):")
for feature, correlation in target_correlations.items():
    strength = "Very Strong" if abs(correlation) > 0.7 else "Strong" if abs(correlation) > 0.4 else "Moderate" if abs(correlation) > 0.2 else "Weak"
    direction = "Positive" if correlation > 0 else "Negative"
    print(f"   • {feature}: r = {correlation:.3f} ({strength} {direction})")

# Detailed feature analysis
print(f"\n📊 Detailed Analysis of Top Predictive Features:")
print("-" * 50)

# Analyze top 3 features
top_3_features = target_correlations.head(3).index.tolist()

for feature in top_3_features:
    print(f"\n🔬 {feature} Analysis:")
    
    # Statistical comparison between groups
    no_diabetes = df[df['Outcome'] == 0][feature]
    diabetes = df[df['Outcome'] == 1][feature]
    
    print(f"   No Diabetes - Mean: {no_diabetes.mean():.2f}, Std: {no_diabetes.std():.2f}")
    print(f"   Diabetes    - Mean: {diabetes.mean():.2f}, Std: {diabetes.std():.2f}")
    print(f"   Difference:   {diabetes.mean() - no_diabetes.mean():.2f}")
    
    # Create distribution plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(no_diabetes, alpha=0.7, label='No Diabetes', color='lightgreen', bins=30)
    plt.hist(diabetes, alpha=0.7, label='Diabetes', color='lightcoral', bins=30)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'{feature} Distribution by Diabetes Status')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.violinplot(data=df, x='Outcome', y=feature)
    plt.xlabel('Diabetes Status')
    plt.ylabel(feature)
    plt.title(f'{feature} Violin Plot')
    plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. MODEL SELECTION AND JUSTIFICATION
# ============================================================================

print("\n4️⃣ MODEL SELECTION AND JUSTIFICATION")
print("-" * 50)

print("\n🤖 MACHINE LEARNING MODEL ANALYSIS:")
print("=" * 40)

# Data characteristics analysis
print("\n📊 Dataset Characteristics:")
print(f"   • Sample Size: {len(df)} patients")
print(f"   • Feature Count: {len(df.columns)-1} features")
print(f"   • Target Type: Binary classification (0/1)")
print(f"   • Class Balance: {outcome_counts[0]/len(df)*100:.1f}% / {outcome_counts[1]/len(df)*100:.1f}%")
print(f"   • Missing Values: {df.isnull().sum().sum()}")
print(f"   • Data Types: All numerical features")

# Feature analysis for model selection
print(f"\n🔍 Feature Analysis for Model Selection:")
print(f"   • Strongest Predictor: {target_correlations.index[0]} (r = {target_correlations.iloc[0]:.3f})")
print(f"   • Number of Strong Predictors (|r| > 0.3): {sum(abs(target_correlations) > 0.3)}")
print(f"   • Feature Relationships: {len(strong_correlations)} strong inter-feature correlations")

# Model recommendations
print(f"\n🎯 RECOMMENDED MODELS:")
print("=" * 30)

models_analysis = {
    "Random Forest": {
        "pros": [
            "Handles non-linear relationships well",
            "Built-in feature importance",
            "Robust to outliers",
            "Good with mixed data types",
            "Reduces overfitting through averaging"
        ],
        "cons": [
            "Can be prone to overfitting with small datasets",
            "Less interpretable than single trees"
        ],
        "suitability": "EXCELLENT",
        "score": 9.5
    },
    "Logistic Regression": {
        "pros": [
            "Highly interpretable coefficients",
            "Fast training and prediction",
            "Probabilistic output",
            "Good baseline model",
            "Works well with linear relationships"
        ],
        "cons": [
            "Assumes linear relationship",
            "Sensitive to outliers",
            "May underperform with complex patterns"
        ],
        "suitability": "GOOD",
        "score": 8.0
    },
    "Support Vector Machine": {
        "pros": [
            "Effective with small datasets",
            "Memory efficient",
            "Versatile (different kernels)",
            "Good generalization"
        ],
        "cons": [
            "Sensitive to feature scaling",
            "No probabilistic output",
            "Less interpretable"
        ],
        "suitability": "GOOD",
        "score": 7.5
    },
    "Gradient Boosting": {
        "pros": [
            "Excellent predictive performance",
            "Handles non-linear patterns",
            "Feature importance available",
            "Robust to outliers"
        ],
        "cons": [
            "Prone to overfitting",
            "Requires hyperparameter tuning",
            "Longer training time"
        ],
        "suitability": "VERY GOOD",
        "score": 8.5
    }
}

for model, analysis in models_analysis.items():
    print(f"\n🔸 {model} (Suitability: {analysis['suitability']}, Score: {analysis['score']}/10)")
    print(f"   ✅ Pros:")
    for pro in analysis['pros']:
        print(f"      • {pro}")
    print(f"   ⚠️  Cons:")
    for con in analysis['cons']:
        print(f"      • {con}")

# Final recommendation
print(f"\n🏆 FINAL MODEL RECOMMENDATION:")
print("=" * 35)
print(f"PRIMARY CHOICE: Random Forest Classifier")
print(f"\n🎯 JUSTIFICATION:")
print(f"   1. DATASET SIZE: 768 samples is adequate for Random Forest")
print(f"   2. FEATURE COMPLEXITY: Can capture non-linear relationships between features")
print(f"   3. ROBUSTNESS: Less sensitive to outliers and missing values")
print(f"   4. INTERPRETABILITY: Provides feature importance rankings")
print(f"   5. PERFORMANCE: Expected accuracy 75-85% based on feature correlations")
print(f"   6. OVERFITTING PROTECTION: Ensemble method reduces overfitting risk")
print(f"   7. MEDICAL DOMAIN: Widely used and trusted in healthcare applications")

print(f"\n📊 EXPECTED PERFORMANCE:")
print(f"   • Accuracy Target: >75% (achievable)")
print(f"   • Expected Range: 75-85%")
print(f"   • Cross-validation: Will use 5-fold CV for robust evaluation")
print(f"   • Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC")

print(f"\n🔄 BACKUP OPTIONS:")
print(f"   1. Gradient Boosting (XGBoost/LightGBM) - if Random Forest underperforms")
print(f"   2. Logistic Regression - for baseline comparison and interpretability")
print(f"   3. Ensemble (Voting Classifier) - combining multiple models")

print(f"\n✅ PHASE 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("🚀 Ready for Phase 3: Model Development and Training")
print("=" * 60)

# Summary statistics
print(f"\n📋 PHASE 2 SUMMARY:")
print(f"   ✅ Visualization tools configured")
print(f"   ✅ Feature/feature relationships analyzed ({len(strong_correlations)} strong correlations)")
print(f"   ✅ Feature/target relationships visualized")
print(f"   ✅ Model selected and justified (Random Forest)")
print(f"   📊 Key finding: {target_correlations.index[0]} is strongest predictor (r={target_correlations.iloc[0]:.3f})")
print(f"   🎯 Next phase: Implement Random Forest with hyperparameter tuning")
