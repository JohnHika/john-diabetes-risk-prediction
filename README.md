# Diabetes Risk Prediction Project

## 🎯 Project Objective
Develop a machine learning-based prediction system to identify individuals at high risk of developing diabetes using easily obtainable health metrics, enabling early intervention and prevention strategies.

## 📊 Problem Statement
Diabetes affects over 422 million people worldwide. This project aims to create an accessible screening tool that can:
- Predict diabetes risk with >75% accuracy
- Identify key risk factors for clinical decision-making
- Provide early detection capabilities for healthcare providers
- Support preventive healthcare initiatives

## 🗂️ Project Structure
```
diabetes_risk_prediction/
├── PROJECT_OVERVIEW.md          # Comprehensive project documentation
├── ANALYSIS_REPORT.md           # Detailed analysis results
├── diabetes_analysis.py         # Main analysis script
├── app.py                      # Streamlit web application
├── diabetes.csv                # Original dataset
├── diabetes_cleaned.csv        # Preprocessed dataset
├── diabetes_engineered.csv     # Feature engineered dataset
├── diabetes_lab_phase1.ipynb   # Jupyter notebook analysis
├── requirements.txt            # Python dependencies
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
└── STREAMLIT_DEPLOY.md         # Streamlit deployment guide
```

## 📈 Key Results
- **Best Model**: Random Forest Classifier
- **Accuracy**: 75.97%
- **Most Important Feature**: Glucose (27.6%)
- **Dataset**: 768 patients from Pima Indians Diabetes Database
- **Deployment**: Functional Streamlit web application

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/JohnHika/john-diabetes-risk-prediction.git
cd john-diabetes-risk-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Analysis
```bash
python diabetes_analysis.py
```

### 4. Launch Web Application
```bash
streamlit run app.py
```

## 📋 Features Analyzed
1. **Pregnancies** - Number of pregnancies
2. **Glucose** - Plasma glucose concentration
3. **BloodPressure** - Diastolic blood pressure
4. **SkinThickness** - Triceps skinfold thickness
5. **Insulin** - 2-Hour serum insulin
6. **BMI** - Body Mass Index
7. **DiabetesPedigreeFunction** - Genetic predisposition
8. **Age** - Age in years

## 🏆 Model Performance Comparison
| Model | Accuracy | Status |
|-------|----------|--------|
| **Random Forest** | **75.97%** | **🏆 Best** |
| SVM | 75.32% | Very Good |
| Decision Tree | 72.73% | Good |
| Logistic Regression | 71.43% | Baseline |

## 📊 Key Insights
- **Glucose levels** are the most critical predictor (27.6% importance)
- **BMI and Age** are significant secondary risk factors
- **Family history** (DiabetesPedigreeFunction) plays an important role
- Model achieves clinical-grade accuracy for screening purposes

## 🔬 Methodology
1. **Data Exploration**: Comprehensive EDA with 768 patient records
2. **Data Preprocessing**: Handling zero values and feature scaling
3. **Model Development**: Tested 4 different algorithms
4. **Model Evaluation**: Used accuracy, precision, recall, and F1-score
5. **Feature Analysis**: Identified most important predictive factors
6. **Deployment**: Created user-friendly web application

## 📚 Documentation
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Complete project planning and methodology
- **[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** - Detailed analysis results and findings
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Instructions for deploying the application

## 🎓 Learning Outcomes
This project demonstrates:
- Complete data science workflow from problem definition to deployment
- Supervised machine learning for healthcare applications
- Model comparison and selection techniques
- Feature importance analysis and interpretation
- Web application development for machine learning models

## 📞 Contact
**John Hika**
- GitHub: [@JohnHika](https://github.com/JohnHika)
- Project Repository: [john-diabetes-risk-prediction](https://github.com/JohnHika/john-diabetes-risk-prediction)

---
*This project addresses real-world healthcare challenges using machine learning and follows industry best practices for data science project development.*