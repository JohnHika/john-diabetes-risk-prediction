# Diabetes Risk Prediction - Phase 1

## 🎯 Project Overview

This repository contains the first phase of a comprehensive diabetes risk prediction project using machine learning. The goal is to develop a binary classification model that can predict diabetes risk in patients based on health indicators.

## 📊 Dataset

**Source:** Pima Indians Diabetes Database  
**Origin:** National Institute of Diabetes and Digestive and Kidney Diseases  
**Size:** 768 patient records  
**Features:** 8 predictive features + 1 target variable  

### Features Description:
- **Pregnancies:** Number of times pregnant
- **Glucose:** Plasma glucose concentration (mg/dL)
- **BloodPressure:** Diastolic blood pressure (mm Hg)
- **SkinThickness:** Triceps skin fold thickness (mm)
- **Insulin:** 2-Hour serum insulin (mu U/ml)
- **BMI:** Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction:** Diabetes pedigree function (genetic predisposition)
- **Age:** Age in years
- **Outcome:** Class variable (0 = No diabetes, 1 = Diabetes)

## 🔧 Phase 1 Objectives

✅ **Project Definition** - Established clear classification objectives  
✅ **Data Acquisition** - Loaded and validated the dataset  
✅ **Data Exploration** - Comprehensive EDA and statistical analysis  
✅ **Data Quality Assessment** - Identified and documented data issues  
✅ **Missing Value Treatment** - Replaced zeros with NaN and applied median imputation  
✅ **Outlier Detection** - IQR-based detection with percentile capping  
✅ **Feature Engineering** - Created 7 meaningful derived features  
✅ **Data Export** - Saved cleaned datasets for next phases  

## ⚙️ Feature Engineering

Created 7 new features to enhance predictive power:

1. **BMI_Category** - WHO classification (Underweight, Normal, Overweight, Obese)
2. **Age_Group** - Life stage grouping (Young, Middle-aged, Senior)
3. **Glucose_Category** - Medical thresholds (Normal, Pre-diabetic, Diabetic)
4. **BP_Category** - AHA guidelines (Normal, Elevated, High)
5. **High_Pregnancies** - Risk indicator for ≥5 pregnancies
6. **Risk_Score** - Composite normalized risk metric
7. **Insulin_Resistance** - Metabolic indicator based on glucose+insulin levels

## 📁 File Structure

```
diabetes-risk-prediction/
├── diabetes_lab_phase1.ipynb          # Main analysis notebook
├── diabetes.csv                       # Original dataset
├── diabetes_cleaned.csv               # Basic cleaned dataset
├── diabetes_engineered.csv            # Full preprocessed dataset
├── diabetes_numerical_only.csv        # Numerical features only
├── README.md                          # This documentation
└── requirements.txt                   # Dependencies
```

## 🛠️ Technical Stack

- **Python 3.12+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **scipy** - Statistical functions

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[username]/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook diabetes_lab_phase1.ipynb
   ```

## 📈 Key Results

- **Data Completeness:** 100% (no missing values after preprocessing)
- **Feature Coverage:** 87.5% increase in features (8 → 15)
- **Record Preservation:** 100% (no data loss)
- **Target Balance:** Maintained (65.1% No Diabetes, 34.9% Diabetes)

## 🔮 Next Steps (Phase 2)

1. Advanced exploratory data analysis
2. Feature selection and correlation analysis
3. Model selection and training
4. Cross-validation and hyperparameter tuning
5. Model evaluation and comparison
6. Performance metrics analysis

## 📋 Quality Assurance

- Comprehensive data validation
- Statistical outlier treatment
- Feature engineering with domain knowledge
- Data integrity verification
- Reproducible preprocessing pipeline

## 👥 Contributing

This is an academic project. For suggestions or improvements, please open an issue or submit a pull request.

## 📄 License

This project is for educational purposes. Dataset credit goes to the National Institute of Diabetes and Digestive and Kidney Diseases.

---

**Phase 1 Status:** ✅ Complete  
**Next Phase:** Model Development and Evaluation  
**Last Updated:** July 28, 2025
