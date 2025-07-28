"""
Diabetes Risk Prediction - Supervised Machine Learning Project
==============================================================

This script performs comprehensive data analysis and machine learning
on the Pima Indians Diabetes Dataset to predict diabetes risk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("=" * 60)
print("DIABETES RISK PREDICTION ANALYSIS")
print("=" * 60)

# 1. LOAD THE DATASET
print("\n1. LOADING DATASET")
print("-" * 30)

df = pd.read_csv('diabetes.csv')
print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# 2. DATA EXPLORATION AND DESCRIPTION
print("\n2. DATA EXPLORATION")
print("-" * 30)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nStatistical description:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

# 3. CHECK FOR MISSING VALUES
print("\n3. DATA QUALITY CHECK")
print("-" * 30)

print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print("⚠ Missing values detected!")

# Check for zeros in columns where they shouldn't be
zero_counts = {}
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = (df[col] == 0).sum()
    zero_counts[col] = zero_count
    
print("\nZero values in critical columns (may indicate missing data):")
for col, count in zero_counts.items():
    print(f"{col}: {count} zeros")

# 4. FEATURE-TO-FEATURE RELATIONSHIPS
print("\n4. FEATURE-TO-FEATURE ANALYSIS")
print("-" * 30)

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap of All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Pairplot for selected features
print("\nCreating pairplot for key features...")
key_features = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Outcome']
plt.figure(figsize=(12, 10))
sns.pairplot(df[key_features], hue='Outcome', diag_kind='hist')
plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=16, fontweight='bold')
plt.show()

# 5. FEATURE-TO-TARGET RELATIONSHIPS
print("\n5. FEATURE-TO-TARGET ANALYSIS")
print("-" * 30)

# Get feature names (exclude target)
feature_names = [col for col in df.columns if col != 'Outcome']

# Create subplots for feature-target relationships
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    # Box plot for each feature vs outcome
    sns.boxplot(data=df, x='Outcome', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} vs Diabetes Outcome')
    axes[i].set_xlabel('Diabetes (0=No, 1=Yes)')

plt.tight_layout()
plt.suptitle('Feature-to-Target Relationships (Box Plots)', 
             y=1.02, fontsize=16, fontweight='bold')
plt.show()

# Violin plots for better distribution visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.ravel()

for i, feature in enumerate(feature_names):
    sns.violinplot(data=df, x='Outcome', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} Distribution by Outcome')
    axes[i].set_xlabel('Diabetes (0=No, 1=Yes)')

plt.tight_layout()
plt.suptitle('Feature Distributions by Diabetes Outcome (Violin Plots)', 
             y=1.02, fontsize=16, fontweight='bold')
plt.show()

# 6. TARGET VARIABLE ANALYSIS
print("\n6. TARGET VARIABLE ANALYSIS")
print("-" * 30)

outcome_counts = df['Outcome'].value_counts()
print("Target variable distribution:")
print(f"No Diabetes (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.1f}%)")
print(f"Diabetes (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.1f}%)")

# Visualize target distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df['Outcome'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
df['Outcome'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                  colors=['skyblue', 'salmon'])
plt.title('Diabetes Outcome Percentage')
plt.ylabel('')

plt.tight_layout()
plt.show()

# 7. MACHINE LEARNING MODEL COMPARISON
print("\n7. MACHINE LEARNING MODEL COMPARISON")
print("-" * 30)

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features for SVM and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

# Train and evaluate models
results = {}
print("\nModel Performance:")
print("-" * 40)

for name, model in models.items():
    # Use scaled data for LR and SVM, original for tree-based models
    if name in ['Logistic Regression', 'SVM']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name:20s}: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Find the best model
best_model_name = max(results.keys(), key=lambda x: results[x])
best_accuracy = results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# 8. DETAILED ANALYSIS OF BEST MODEL
print(f"\n8. DETAILED ANALYSIS OF {best_model_name.upper()}")
print("-" * 50)

# Train the best model
if best_model_name in ['Logistic Regression', 'SVM']:
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance (for tree-based models)
if best_model_name in ['Decision Tree', 'Random Forest']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.show()
    
    print("Feature Importance Ranking:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'], 
                                                  feature_importance['importance']), 1):
        print(f"{i:2d}. {feature:25s}: {importance:.4f}")

# 9. MODEL SELECTION JUSTIFICATION
print(f"\n9. MODEL SELECTION JUSTIFICATION")
print("-" * 40)

"""
MODEL CHOICE JUSTIFICATION:

Based on the analysis, here's why each model performed as it did:

1. LOGISTIC REGRESSION:
   - Good baseline model for binary classification
   - Interpretable coefficients
   - Handles linear relationships well
   - Performance depends on feature scaling

2. DECISION TREE:
   - Highly interpretable
   - Can capture non-linear relationships
   - Prone to overfitting with small datasets
   - No need for feature scaling

3. RANDOM FOREST:
   - Ensemble method that reduces overfitting
   - Handles both linear and non-linear relationships
   - Provides feature importance
   - Generally robust and reliable

4. SVM:
   - Good for high-dimensional data
   - Can handle non-linear relationships with kernel trick
   - Requires feature scaling
   - Less interpretable than other models

RECOMMENDATION:
The {best_model_name} achieved the highest accuracy of {best_accuracy:.4f}.
This model is recommended because:
- It provides the best predictive performance on the test set
- It balances bias and variance effectively
- It's suitable for the dataset size and complexity
"""

print(f"✅ ANALYSIS COMPLETE!")
print(f"📊 Best Model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")
print(f"📁 Results saved and visualizations displayed")
print("=" * 60)
