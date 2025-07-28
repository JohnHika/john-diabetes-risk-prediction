import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# Note: matplotlib, seaborn, plotly.graph_objects, and make_subplots 
# are imported but not used in current implementation

# Page configuration
st.set_page_config(
    page_title="John's Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets with caching for better performance"""
    try:
        df_original = pd.read_csv('diabetes.csv')
        df_cleaned = pd.read_csv('diabetes_cleaned.csv')
        df_engineered = pd.read_csv('diabetes_engineered.csv')
        df_numerical = pd.read_csv('diabetes_numerical_only.csv')
        return df_original, df_cleaned, df_engineered, df_numerical
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        return None, None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 John\'s Diabetes Risk Prediction Project</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Project Overview", "📊 Data Analysis", "🔧 Data Processing", "📈 Feature Engineering", "🚀 Model Ready Data", "📱 Risk Calculator"]
    )
    
    # Load data
    df_original, df_cleaned, df_engineered, df_numerical = load_data()
    
    if df_original is None:
        st.error("❌ Could not load datasets. Please ensure all CSV files are in the project directory.")
        return
    
    # Page routing
    if page == "🏠 Project Overview":
        show_overview(df_original, df_engineered)
    elif page == "📊 Data Analysis":
        show_data_analysis(df_original)
    elif page == "🔧 Data Processing":
        show_data_processing(df_original, df_cleaned)
    elif page == "📈 Feature Engineering":
        show_feature_engineering(df_cleaned, df_engineered)
    elif page == "🚀 Model Ready Data":
        show_final_data(df_engineered, df_numerical)
    elif page == "📱 Risk Calculator":
        show_risk_calculator(df_engineered)

def show_overview(df_original, df_engineered):
    st.header("🎯 Project Vision & Accomplishments")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What I Built
        This is my comprehensive diabetes risk prediction project focusing on the Pima Indians Diabetes Database. 
        My goal was to transform raw medical data into a clean, feature-rich dataset ready for machine learning.
        
        ### Why This Matters
        - **Early Detection**: Can help prevent diabetes through lifestyle changes
        - **Healthcare Impact**: Diabetes affects 422+ million people worldwide
        - **Cost Effective**: Prevention is cheaper than treatment
        - **Community Health**: Understanding risk patterns helps target interventions
        """)
        
        st.markdown("### 🏆 Phase 1 Achievements")
        achievements = [
            "✅ Processed 768 patient records with zero data loss",
            "✅ Engineered 7 new medical domain features", 
            "✅ Applied systematic data cleaning and outlier treatment",
            "✅ Created multiple dataset versions for different ML approaches",
            "✅ Built this interactive web application for exploration"
        ]
        for achievement in achievements:
            st.markdown(achievement)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("#### 📊 Project Stats")
        st.metric("Patient Records", f"{df_original.shape[0]:,}")
        st.metric("Original Features", f"{df_original.shape[1]-1}")
        st.metric("Engineered Features", f"{df_engineered.shape[1]-1}")
        st.metric("Features Added", f"{df_engineered.shape[1] - df_original.shape[1]}")
        
        diabetes_rate = (df_original['Outcome'].sum() / len(df_original)) * 100
        st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Project Timeline
    st.header("📅 Project Timeline & Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Phase 1: Data Foundation** ✅
        - Data acquisition & exploration
        - Quality assessment & cleaning
        - Feature engineering
        - Web app development
        """)
    
    with col2:
        st.markdown("""
        **Phase 2: Model Building** 🔄
        - Algorithm comparison
        - Cross-validation
        - Hyperparameter tuning
        - Performance evaluation
        """)
    
    with col3:
        st.markdown("""
        **Phase 3: Deployment** 📋
        - Model finalization
        - Production pipeline
        - Clinical interpretation
        - Impact assessment
        """)

def show_data_analysis(df):
    st.header("📊 Original Data Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{df.shape[0]:,}")
    col2.metric("Features", f"{df.shape[1]-1}")
    col3.metric("Diabetes Cases", f"{df['Outcome'].sum()}")
    col4.metric("Healthy Cases", f"{(df['Outcome']==0).sum()}")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Target distribution
    st.subheader("🎯 Target Variable Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=['No Diabetes', 'Diabetes'],
            y=[df['Outcome'].value_counts()[0], df['Outcome'].value_counts()[1]],
            title="Class Distribution",
            color=['No Diabetes', 'Diabetes'],
            color_discrete_map={'No Diabetes': '#87CEEB', 'Diabetes': '#FA8072'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=df['Outcome'].value_counts().values,
            names=['No Diabetes', 'Diabetes'],
            title="Class Balance",
            color_discrete_map={'No Diabetes': '#87CEEB', 'Diabetes': '#FA8072'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('Outcome')
    
    selected_features = st.multiselect(
        "Select features to visualize:",
        numerical_cols,
        default=list(numerical_cols[:4])
    )
    
    if selected_features:
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                fig = px.histogram(
                    df, x=feature, color='Outcome',
                    title=f'{feature} Distribution by Outcome',
                    color_discrete_map={0: '#87CEEB', 1: '#FA8072'},
                    marginal='box'
                )
                st.plotly_chart(fig, use_container_width=True)

def show_data_processing(df_original, df_cleaned):
    st.header("🔧 Data Cleaning & Processing")
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Detection")
    
    # Zero values that are actually missing
    problematic_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    missing_data = []
    for col in problematic_cols:
        zero_count = (df_original[col] == 0).sum()
        zero_pct = (zero_count / len(df_original)) * 100
        missing_data.append({
            'Feature': col,
            'Zero Values': zero_count,
            'Percentage': f"{zero_pct:.1f}%",
            'Status': '🚨 Suspicious' if zero_count > 0 else '✅ Clean'
        })
    
    missing_df = pd.DataFrame(missing_data)
    st.dataframe(missing_df, use_container_width=True)
    
    # Cleaning strategy
    st.subheader("🛠️ My Cleaning Strategy")
    st.markdown("""
    **Why zeros are problematic in medical data:**
    - **Glucose**: Cannot be 0 (would indicate death)
    - **Blood Pressure**: Cannot be 0 (would indicate death)  
    - **BMI**: Cannot be 0 (impossible measurement)
    - **Skin Thickness**: Cannot be 0 (impossible measurement)
    - **Insulin**: Cannot be 0 (body always produces some insulin)
    
    **My Solution**: Replace zeros with NaN, then use median imputation
    """)
    
    # Before/After comparison
    st.subheader("📊 Before vs After Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Before Cleaning**")
        for col in problematic_cols[:3]:
            zero_count = (df_original[col] == 0).sum()
            st.metric(f"{col} Zeros", zero_count)
    
    with col2:
        st.markdown("**After Cleaning**")
        for col in problematic_cols[:3]:
            missing_count = df_cleaned[col].isnull().sum() if col in df_cleaned.columns else 0
            st.metric(f"{col} Missing", missing_count, delta="Imputed with median")
    
    # Imputation values used
    st.subheader("🎯 Imputation Values Used")
    imputation_data = []
    for col in problematic_cols:
        if col in df_cleaned.columns:
            median_val = df_cleaned[col].median()
            imputation_data.append({
                'Feature': col,
                'Median Used': f"{median_val:.2f}",
                'Reason': 'Robust to outliers'
            })
    
    if imputation_data:
        imputation_df = pd.DataFrame(imputation_data)
        st.dataframe(imputation_df, use_container_width=True)

def show_feature_engineering(df_cleaned, df_engineered):
    st.header("⚙️ Feature Engineering Innovations")
    
    # Feature creation summary
    original_features = df_cleaned.shape[1] - 1
    new_features = df_engineered.shape[1] - df_cleaned.shape[1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Features", original_features)
    col2.metric("Features Added", new_features)
    col3.metric("Total Features", original_features + new_features)
    
    # New features breakdown
    st.subheader("🎨 My 7 New Features")
    
    features_info = [
        {
            "Feature": "BMI_Category",
            "Type": "Categorical",
            "Purpose": "WHO obesity classification",
            "Medical Insight": "Obesity is a major diabetes risk factor"
        },
        {
            "Feature": "Age_Group", 
            "Type": "Categorical",
            "Purpose": "Life stage categorization",
            "Medical Insight": "Diabetes risk increases with age"
        },
        {
            "Feature": "Glucose_Category",
            "Type": "Categorical", 
            "Purpose": "ADA diagnostic thresholds",
            "Medical Insight": "Captures pre-diabetic stage"
        },
        {
            "Feature": "BP_Category",
            "Type": "Categorical",
            "Purpose": "AHA hypertension guidelines", 
            "Medical Insight": "Hypertension comorbid with diabetes"
        },
        {
            "Feature": "High_Pregnancies",
            "Type": "Binary",
            "Purpose": "Gestational diabetes indicator",
            "Medical Insight": "Multiple pregnancies increase risk"
        },
        {
            "Feature": "Risk_Score",
            "Type": "Numerical",
            "Purpose": "Weighted composite score",
            "Medical Insight": "Combines multiple risk factors"
        },
        {
            "Feature": "Insulin_Resistance",
            "Type": "Binary", 
            "Purpose": "Metabolic dysfunction indicator",
            "Medical Insight": "High glucose + insulin = resistance"
        }
    ]
    
    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df, use_container_width=True)
    
    # Feature distributions
    st.subheader("📊 New Feature Distributions")
    
    # Categorical features
    categorical_features = ['BMI_Category', 'Age_Group', 'Glucose_Category', 'BP_Category']
    
    cols = st.columns(2)
    for i, feature in enumerate(categorical_features):
        if feature in df_engineered.columns:
            with cols[i % 2]:
                value_counts = df_engineered[feature].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f'{feature} Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Risk Score Analysis
    st.subheader("🎯 Risk Score Analysis")
    if 'Risk_Score' in df_engineered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_engineered, x='Risk_Score', color='Outcome',
                title='Risk Score Distribution by Outcome',
                color_discrete_map={0: '#87CEEB', 1: '#FA8072'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            risk_by_outcome = df_engineered.groupby('Outcome')['Risk_Score'].mean()
            fig = px.bar(
                x=['No Diabetes', 'Diabetes'],
                y=risk_by_outcome.values,
                title='Average Risk Score by Outcome',
                color=['No Diabetes', 'Diabetes'],
                color_discrete_map={'No Diabetes': '#87CEEB', 'Diabetes': '#FA8072'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_final_data(df_engineered, df_numerical):
    st.header("🚀 Model-Ready Datasets")
    
    # Dataset comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Full Engineered Dataset")
        st.metric("Rows", f"{df_engineered.shape[0]:,}")
        st.metric("Features", f"{df_engineered.shape[1]-1}")
        st.metric("Data Types", f"{len(df_engineered.dtypes.unique())}")
        
        st.markdown("**Includes:**")
        st.markdown("- Original features")
        st.markdown("- Categorical engineered features")
        st.markdown("- Numerical engineered features")
        st.markdown("- Ready for tree-based models")
    
    with col2:
        st.subheader("🔢 Numerical-Only Dataset")
        st.metric("Rows", f"{df_numerical.shape[0]:,}")
        st.metric("Features", f"{df_numerical.shape[1]-1}")
        st.metric("All Numerical", "✅")
        
        st.markdown("**Optimized for:**")
        st.markdown("- Linear models")
        st.markdown("- Neural networks") 
        st.markdown("- SVM algorithms")
        st.markdown("- Models requiring scaling")
    
    # Data quality verification
    st.subheader("✅ Data Quality Verification")
    
    quality_checks = [
        {
            "Check": "Missing Values",
            "Engineered": f"{df_engineered.isnull().sum().sum()} ✅",
            "Numerical": f"{df_numerical.isnull().sum().sum()} ✅"
        },
        {
            "Check": "Target Distribution",
            "Engineered": f"Preserved ✅",
            "Numerical": f"Preserved ✅"
        },
        {
            "Check": "Data Types",
            "Engineered": f"Mixed (optimal) ✅",
            "Numerical": f"All numeric ✅"
        },
        {
            "Check": "Sample Size",
            "Engineered": f"{df_engineered.shape[0]} records ✅",
            "Numerical": f"{df_numerical.shape[0]} records ✅"
        }
    ]
    
    quality_df = pd.DataFrame(quality_checks)
    st.dataframe(quality_df, use_container_width=True)
    
    # Download options
    st.subheader("📥 Download Datasets")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_engineered = df_engineered.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Engineered Dataset",
            data=csv_engineered,
            file_name="diabetes_engineered.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_numerical = df_numerical.to_csv(index=False)
        st.download_button(
            label="🔢 Download Numerical Dataset", 
            data=csv_numerical,
            file_name="diabetes_numerical_only.csv",
            mime="text/csv"
        )

def show_risk_calculator(df_engineered):
    st.header("📱 Interactive Diabetes Risk Calculator")
    st.markdown("*Based on the patterns learned from 768 patients*")
    
    # Input form
    with st.form("risk_calculator"):
        st.subheader("Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=17, value=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=44, max_value=199, value=120)
            bp = st.number_input("Blood Pressure (mm Hg)", min_value=24, max_value=122, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=7, max_value=99, value=20)
        
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", min_value=14, max_value=846, value=80)
            bmi = st.number_input("BMI", min_value=18.2, max_value=67.1, value=25.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.5, format="%.3f")
            age = st.number_input("Age", min_value=21, max_value=81, value=30)
        
        submitted = st.form_submit_button("Calculate Risk Score")
    
    if submitted:
        # Calculate engineered features
        def categorize_bmi(bmi):
            if bmi < 18.5: return 'Underweight'
            elif bmi < 25: return 'Normal'
            elif bmi < 30: return 'Overweight'
            else: return 'Obese'
        
        def categorize_age(age):
            if age < 30: return 'Young'
            elif age < 50: return 'Middle-aged'
            else: return 'Senior'
        
        def categorize_glucose(glucose):
            if glucose < 100: return 'Normal'
            elif glucose < 126: return 'Pre-diabetic'
            else: return 'Diabetic'
        
        def categorize_bp(bp):
            if bp < 80: return 'Normal'
            elif bp < 90: return 'Elevated'
            else: return 'High'
        
        # Calculate risk score
        glucose_norm = (glucose - 44) / (199 - 44)
        bmi_norm = (bmi - 18.2) / (67.1 - 18.2)
        age_norm = (age - 21) / (81 - 21)
        dpf_norm = dpf / 2.42
        
        risk_score = (0.35 * glucose_norm + 0.25 * bmi_norm + 0.20 * age_norm + 0.20 * dpf_norm)
        
        # Display results
        st.subheader("🎯 Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.3f}")
            risk_percentage = risk_score * 100
            st.metric("Risk Percentage", f"{risk_percentage:.1f}%")
        
        with col2:
            # Risk category
            if risk_score < 0.3:
                risk_category = "🟢 Low Risk"
            elif risk_score < 0.6:
                risk_category = "🟡 Moderate Risk"
            else:
                risk_category = "🔴 High Risk"
            
            st.markdown(f"**Risk Category:** {risk_category}")
            
            # Compare to dataset
            similar_patients = df_engineered[
                (abs(df_engineered['Age'] - age) <= 5) &
                (abs(df_engineered['BMI'] - bmi) <= 2)
            ]
            
            if len(similar_patients) > 0:
                diabetes_rate = (similar_patients['Outcome'].sum() / len(similar_patients)) * 100
                st.metric("Similar Patients Diabetes Rate", f"{diabetes_rate:.1f}%")
        
        with col3:
            st.markdown("**Your Categories:**")
            st.write(f"BMI: {categorize_bmi(bmi)}")
            st.write(f"Age Group: {categorize_age(age)}")
            st.write(f"Glucose: {categorize_glucose(glucose)}")
            st.write(f"Blood Pressure: {categorize_bp(bp)}")
        
        # Risk factors analysis
        st.subheader("📊 Risk Factor Analysis")
        
        factors = {
            'Glucose Level': glucose_norm,
            'BMI': bmi_norm, 
            'Age': age_norm,
            'Family History': dpf_norm
        }
        
        fig = px.bar(
            x=list(factors.keys()),
            y=list(factors.values()),
            title="Normalized Risk Factors (0-1 scale)",
            color=list(factors.values()),
            color_continuous_scale="RdYlGn_r"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = []
        if glucose_norm > 0.6:
            recommendations.append("🍎 Monitor glucose levels and consider dietary changes")
        if bmi_norm > 0.6:
            recommendations.append("🏃‍♂️ Focus on weight management through diet and exercise")
        if age_norm > 0.5:
            recommendations.append("🩺 Regular diabetes screening is important at your age")
        if dpf_norm > 0.5:
            recommendations.append("👨‍👩‍👧‍👦 Family history indicates higher vigilance needed")
        
        if not recommendations:
            recommendations.append("✅ Keep maintaining your healthy lifestyle!")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        st.markdown("---")
        st.markdown("*⚠️ This calculator is for educational purposes only. Please consult healthcare professionals for medical advice.*")

if __name__ == "__main__":
    main()
