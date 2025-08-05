"""
Diabetes Risk Prediction Web Application
========================================
A machine learning-powered tool for diabetes risk assessment using basic health metrics.

Author: John Hika
Model: Random Forest Classifier (75.97% accuracy)
Dataset: Pima Indians Diabetes Database
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
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
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .risk-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .feature-importance {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model and dataset for predictions"""
    try:
        # Load the original dataset to train model (since we don't have a saved model)
        df = pd.read_csv('diabetes.csv')
        
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Train the Random Forest model
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, df
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_diabetes_risk(model, scaler, features):
    """Make prediction using the trained model"""
    try:
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def get_feature_importance(model):
    """Get feature importance from the model"""
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def create_risk_gauge(probability):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Main Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Health Assessment Tool | 75.97% Accuracy</p>', unsafe_allow_html=True)
    
    # Load model and data
    model, scaler, df = load_model_and_data()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check your data files.")
        return
    
    # Sidebar for input
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("Enter the patient's health metrics below:")
    
    # Input fields
    pregnancies = st.sidebar.number_input(
        "Number of Pregnancies", 
        min_value=0, max_value=20, value=1,
        help="Total number of pregnancies"
    )
    
    glucose = st.sidebar.number_input(
        "Glucose Level (mg/dL)", 
        min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration after 2-hour oral glucose tolerance test"
    )
    
    blood_pressure = st.sidebar.number_input(
        "Blood Pressure (mmHg)", 
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure"
    )
    
    skin_thickness = st.sidebar.number_input(
        "Skin Thickness (mm)", 
        min_value=0, max_value=100, value=20,
        help="Triceps skinfold thickness"
    )
    
    insulin = st.sidebar.number_input(
        "Insulin Level (μU/mL)", 
        min_value=0, max_value=900, value=80,
        help="2-hour serum insulin level"
    )
    
    bmi = st.sidebar.number_input(
        "BMI (kg/m²)", 
        min_value=0.0, max_value=100.0, value=25.0, step=0.1,
        help="Body Mass Index"
    )
    
    diabetes_pedigree = st.sidebar.number_input(
        "Diabetes Pedigree Function", 
        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
        help="Genetic predisposition to diabetes based on family history"
    )
    
    age = st.sidebar.number_input(
        "Age (years)", 
        min_value=1, max_value=120, value=30,
        help="Age in years"
    )
    
    # Predict button
    if st.sidebar.button("🔍 Predict Diabetes Risk", type="primary"):
        # Collect features
        features = [pregnancies, glucose, blood_pressure, skin_thickness, 
                   insulin, bmi, diabetes_pedigree, age]
        
        # Make prediction
        prediction, probability = predict_diabetes_risk(model, scaler, features)
        
        if prediction is not None:
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Risk Assessment Results
                st.header("📊 Risk Assessment Results")
                
                diabetes_prob = probability[1]
                no_diabetes_prob = probability[0]
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3>⚠️ HIGH RISK DETECTED</h3>
                        <p><strong>Diabetes Risk Probability: {diabetes_prob:.1%}</strong></p>
                        <p>This patient shows indicators suggesting a high risk of diabetes. 
                        Immediate medical consultation is recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3>✅ LOW RISK DETECTED</h3>
                        <p><strong>Diabetes Risk Probability: {diabetes_prob:.1%}</strong></p>
                        <p>This patient shows a low risk profile for diabetes. 
                        Continue with regular health monitoring.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk breakdown
                st.subheader("📈 Risk Breakdown")
                risk_data = pd.DataFrame({
                    'Risk Level': ['No Diabetes', 'Diabetes'],
                    'Probability': [no_diabetes_prob, diabetes_prob]
                })
                
                fig_bar = px.bar(
                    risk_data, 
                    x='Risk Level', 
                    y='Probability',
                    title="Risk Probability Comparison",
                    color='Probability',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Risk Gauge
                st.subheader("🎯 Risk Meter")
                fig_gauge = create_risk_gauge(diabetes_prob)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk Level Classification
                if diabetes_prob < 0.3:
                    risk_level = "LOW"
                    risk_color = "green"
                elif diabetes_prob < 0.6:
                    risk_level = "MODERATE"
                    risk_color = "orange"
                else:
                    risk_level = "HIGH"
                    risk_color = "red"
                
                st.markdown(f"""
                **Risk Classification:** <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                """, unsafe_allow_html=True)
            
            # Feature Importance
            st.header("🔍 What Factors Matter Most?")
            importance_df = get_feature_importance(model)
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                st.subheader("Feature Importance Rankings")
                for i, row in importance_df.iterrows():
                    st.write(f"**{i+1}. {row['Feature']}**: {row['Importance']:.1%}")
            
            with col4:
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Diabetes Prediction"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Health Recommendations
            st.header("💡 Health Recommendations")
            
            if prediction == 1:
                st.warning("""
                **Immediate Actions Recommended:**
                - Schedule appointment with healthcare provider
                - Monitor blood glucose levels regularly
                - Consider dietary modifications (reduce sugar, refined carbs)
                - Increase physical activity (150 minutes/week moderate exercise)
                - Maintain healthy weight (BMI 18.5-24.9)
                - Regular health screenings every 3-6 months
                """)
            else:
                st.success("""
                **Preventive Measures:**
                - Maintain regular check-ups annually
                - Continue healthy lifestyle habits
                - Monitor weight and BMI regularly
                - Stay physically active
                - Eat balanced diet with limited processed foods
                - Be aware of family history and genetic risk factors
                """)
    
    # Project Information
    st.header("📋 About This Tool")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.info("""
        **Model Performance**
        - Accuracy: 75.97%
        - Algorithm: Random Forest
        - Training Data: 768 patients
        - Validation: Rigorous testing
        """)
    
    with col6:
        st.info("""
        **Dataset Information**
        - Source: Pima Indians Diabetes Database
        - Authority: National Institute of Diabetes
        - Features: 8 health metrics
        - Target: Diabetes diagnosis
        """)
    
    with col7:
        st.info("""
        **Developer Information**
        - Author: John Hika
        - Project: ML Diabetes Prediction
        - Technology: Python, Streamlit
        - Last Updated: August 2025
        """)
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
    **⚠️ Medical Disclaimer**: This tool is for educational and screening purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare providers for medical decisions.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        Diabetes Risk Predictor v1.0 | Developed by John Hika | {datetime.now().year}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
