"""
Diabetes Risk Prediction - Phase 3: Model Development and Training (Streamlit)
==============================================================================

PROJECT: Diabetes Risk Prediction through Supervised Machine Learning
PHASE: 3 - Model Development and Training with Full Process Accountability
AUTHOR: John Hika
DATE: August 5, 2025

PHASE 3 OBJECTIVES:
1. Prepare your data for training and testing
2. Train your model
3. Evaluate your model with the appropriate evaluation metrics
4. Check if there is an overfitting or underfitting issue and act accordingly
5. Tune the model parameters and repeat until you get the desired results

ACCOUNTABILITY FEATURES:
✅ Real-time training progress tracking
✅ Complete data preparation audit trail
✅ Live model performance monitoring
✅ Overfitting/underfitting detection with evidence
✅ Hyperparameter tuning process documentation
✅ Iteration tracking with performance comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)

import warnings
import time
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Phase 3: Model Development & Training",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for accountability display
st.markdown("""
<style>
.process-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 10px 0;
}

.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 10px 0;
}

.warning-box {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin: 10px 0;
}

.error-box {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #dc3545;
    margin: 10px 0;
}

.metric-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin: 10px 0;
}

.iteration-box {
    background-color: #f3e5f5;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #9c27b0;
    margin: 10px 0;
}

.phase-header {
    color: #1f77b4;
    border-bottom: 3px solid #1f77b4;
    padding-bottom: 10px;
}

.live-log {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    font-family: monospace;
    font-size: 12px;
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for comprehensive tracking
if 'phase3_log' not in st.session_state:
    st.session_state.phase3_log = []
if 'phase3_status' not in st.session_state:
    st.session_state.phase3_status = {
        'data_preparation': 'pending',
        'model_training': 'pending',
        'model_evaluation': 'pending',
        'overfitting_check': 'pending',
        'hyperparameter_tuning': 'pending'
    }
if 'training_iterations' not in st.session_state:
    st.session_state.training_iterations = []
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'target_accuracy' not in st.session_state:
    st.session_state.target_accuracy = 0.75

def log_process(message, status="info", details=None):
    """Enhanced logging with detailed information for accountability"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'message': message,
        'status': status,
        'details': details
    }
    st.session_state.phase3_log.append(log_entry)

def display_live_log():
    """Display comprehensive live process log"""
    st.subheader("🔍 Live Training Process Log")
    
    if st.session_state.phase3_log:
        log_text = ""
        for entry in st.session_state.phase3_log[-25:]:  # Show last 25 entries
            status_icon = {
                'info': 'ℹ️',
                'success': '✅',
                'warning': '⚠️',
                'error': '❌',
                'training': '🔄',
                'evaluation': '📊',
                'tuning': '⚙️'
            }.get(entry['status'], 'ℹ️')
            
            log_text += f"[{entry['timestamp']}] {status_icon} {entry['message']}\n"
            if entry['details']:
                log_text += f"    📋 {entry['details']}\n"
        
        st.markdown(f'<div class="live-log">{log_text}</div>', unsafe_allow_html=True)
    else:
        st.info("Training log is empty. Start the process to see live updates.")

def execute_with_accountability(step_name, step_function, *args, **kwargs):
    """Execute steps with comprehensive accountability and error handling"""
    log_process(f"🚀 Starting: {step_name}", "training")
    
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        status_text.text(f"🔄 Executing: {step_name}")
        progress_bar.progress(25)
        
        # Execute the step with progress tracking
        result = step_function(*args, **kwargs)
        
        progress_bar.progress(75)
        status_text.text(f"⏳ Finalizing: {step_name}")
        
        progress_bar.progress(100)
        status_text.text(f"✅ Completed: {step_name}")
        
        log_process(f"✅ Successfully completed: {step_name}", "success")
        
        # Clear progress indicators
        time.sleep(1)
        progress_container.empty()
        
        return result
        
    except Exception as e:
        progress_container.empty()
        
        error_msg = f"❌ Error in {step_name}: {str(e)}"
        log_process(error_msg, "error", f"Exception details: {type(e).__name__}")
        st.error(error_msg)
        
        with st.expander("🔍 Error Details for Debugging"):
            st.code(f"Error Type: {type(e).__name__}\nError Message: {str(e)}")
        
        return None

def prepare_training_data():
    """Comprehensive data preparation with full accountability"""
    log_process("🔧 Starting data preparation process", "training")
    
    try:
        # Load dataset
        df = pd.read_csv('diabetes.csv')
        log_process(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features", "info")
        
        # Data quality assessment
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        log_process(f"🔍 Data quality check: {missing_values} missing values, {duplicates} duplicates", "info")
        
        # Handle zero values (identified as potential missing data in Phase 1)
        features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        zero_handling_log = {}
        
        for feature in features_with_zeros:
            if feature in df.columns:
                zero_count = (df[feature] == 0).sum()
                if zero_count > 0:
                    # Replace zeros with median for medical features
                    median_value = df[df[feature] > 0][feature].median()
                    df[feature] = df[feature].replace(0, median_value)
                    zero_handling_log[feature] = {
                        'zeros_replaced': zero_count,
                        'replacement_value': median_value
                    }
                    log_process(f"🔧 {feature}: Replaced {zero_count} zeros with median {median_value:.2f}", "info")
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        log_process(f"🎯 Features (X): {X.shape[1]} columns, Target (y): {y.value_counts().to_dict()}", "info")
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        log_process(f"📊 Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples", "success")
        log_process(f"📊 Train class distribution: {y_train.value_counts().to_dict()}", "info")
        log_process(f"📊 Test class distribution: {y_test.value_counts().to_dict()}", "info")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        log_process("🔧 Feature scaling completed using StandardScaler", "success")
        
        preparation_results = {
            'original_data': df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'feature_names': X.columns.tolist(),
            'zero_handling_log': zero_handling_log,
            'data_stats': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(X.columns),
                'class_balance_train': y_train.value_counts(normalize=True).to_dict(),
                'class_balance_test': y_test.value_counts(normalize=True).to_dict()
            }
        }
        
        return preparation_results
        
    except Exception as e:
        log_process(f"❌ Data preparation failed: {str(e)}", "error")
        return None

def train_model(data_prep, model_type="RandomForest", hyperparameters=None):
    """Train model with comprehensive tracking and documentation"""
    log_process(f"🤖 Starting {model_type} model training", "training")
    
    try:
        # Set default hyperparameters
        if hyperparameters is None:
            if model_type == "RandomForest":
                hyperparameters = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            elif model_type == "LogisticRegression":
                hyperparameters = {
                    'C': 1.0,
                    'random_state': 42,
                    'max_iter': 1000
                }
        
        log_process(f"⚙️ Hyperparameters: {hyperparameters}", "info", json.dumps(hyperparameters, indent=2))
        
        # Initialize model
        if model_type == "RandomForest":
            model = RandomForestClassifier(**hyperparameters)
        elif model_type == "LogisticRegression":
            model = LogisticRegression(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Record training start
        training_start = time.time()
        log_process(f"🔄 Training {model_type} model...", "training")
        
        # Train the model
        model.fit(data_prep['X_train_scaled'], data_prep['y_train'])
        
        training_time = time.time() - training_start
        log_process(f"✅ Model training completed in {training_time:.2f} seconds", "success")
        
        # Make predictions
        train_predictions = model.predict(data_prep['X_train_scaled'])
        test_predictions = model.predict(data_prep['X_test_scaled'])
        
        # Get probability predictions for ROC analysis
        if hasattr(model, 'predict_proba'):
            train_probabilities = model.predict_proba(data_prep['X_train_scaled'])[:, 1]
            test_probabilities = model.predict_proba(data_prep['X_test_scaled'])[:, 1]
        else:
            train_probabilities = None
            test_probabilities = None
        
        log_process("📊 Predictions generated for train and test sets", "success")
        
        training_results = {
            'model': model,
            'model_type': model_type,
            'hyperparameters': hyperparameters,
            'training_time': training_time,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'train_probabilities': train_probabilities,
            'test_probabilities': test_probabilities,
            'feature_names': data_prep['feature_names']
        }
        
        return training_results
        
    except Exception as e:
        log_process(f"❌ Model training failed: {str(e)}", "error")
        return None

def evaluate_model(training_results, data_prep):
    """Comprehensive model evaluation with detailed metrics"""
    log_process("📊 Starting comprehensive model evaluation", "evaluation")
    
    try:
        model = training_results['model']
        
        # Calculate metrics for both train and test sets
        metrics = {}
        
        for dataset, predictions, probabilities, y_true in [
            ('train', training_results['train_predictions'], training_results['train_probabilities'], data_prep['y_train']),
            ('test', training_results['test_predictions'], training_results['test_probabilities'], data_prep['y_test'])
        ]:
            dataset_metrics = {
                'accuracy': accuracy_score(y_true, predictions),
                'precision': precision_score(y_true, predictions),
                'recall': recall_score(y_true, predictions),
                'f1_score': f1_score(y_true, predictions),
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist()
            }
            
            if probabilities is not None:
                dataset_metrics['roc_auc'] = roc_auc_score(y_true, probabilities)
                fpr, tpr, _ = roc_curve(y_true, probabilities)
                dataset_metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            
            metrics[dataset] = dataset_metrics
            
            log_process(f"📈 {dataset.upper()} metrics - Accuracy: {dataset_metrics['accuracy']:.4f}, "
                       f"Precision: {dataset_metrics['precision']:.4f}, "
                       f"Recall: {dataset_metrics['recall']:.4f}, "
                       f"F1: {dataset_metrics['f1_score']:.4f}", "evaluation")
        
        # Cross-validation for robust evaluation
        cv_scores = cross_val_score(model, data_prep['X_train_scaled'], data_prep['y_train'], cv=5, scoring='accuracy')
        metrics['cross_validation'] = {
            'scores': cv_scores.tolist(),
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        log_process(f"🔄 Cross-validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", "evaluation")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(training_results['feature_names'], model.feature_importances_))
            metrics['feature_importance'] = feature_importance
            
            # Log top 3 most important features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features_msg = ", ".join([f"{feat}: {imp:.3f}" for feat, imp in sorted_features[:3]])
            log_process(f"🎯 Top 3 important features: {top_features_msg}", "evaluation")
        
        # Target achievement check
        test_accuracy = metrics['test']['accuracy']
        target_met = test_accuracy >= st.session_state.target_accuracy
        
        log_process(f"🎯 Target accuracy check: {test_accuracy:.4f} vs target {st.session_state.target_accuracy:.4f} - "
                   f"{'✅ TARGET MET' if target_met else '❌ NEEDS IMPROVEMENT'}", 
                   "success" if target_met else "warning")
        
        evaluation_results = {
            'metrics': metrics,
            'target_met': target_met,
            'target_accuracy': st.session_state.target_accuracy,
            'evaluation_summary': {
                'test_accuracy': test_accuracy,
                'train_accuracy': metrics['train']['accuracy'],
                'cv_mean': metrics['cross_validation']['mean'],
                'cv_std': metrics['cross_validation']['std']
            }
        }
        
        return evaluation_results
        
    except Exception as e:
        log_process(f"❌ Model evaluation failed: {str(e)}", "error")
        return None

def check_overfitting_underfitting(training_results, data_prep, evaluation_results):
    """Comprehensive overfitting/underfitting analysis with visual evidence"""
    log_process("🔍 Analyzing overfitting/underfitting patterns", "evaluation")
    
    try:
        model = training_results['model']
        
        # Compare train vs test performance
        train_acc = evaluation_results['metrics']['train']['accuracy']
        test_acc = evaluation_results['metrics']['test']['accuracy']
        accuracy_gap = train_acc - test_acc
        
        # Generate learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, data_prep['X_train_scaled'], data_prep['y_train'],
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std for learning curves
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Analyze the patterns
        analysis = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_gap': accuracy_gap,
            'learning_curves': {
                'train_sizes': train_sizes.tolist(),
                'train_mean': train_mean.tolist(),
                'train_std': train_std.tolist(),
                'test_mean': test_mean.tolist(),
                'test_std': test_std.tolist()
            }
        }
        
        # Determine overfitting/underfitting status
        if accuracy_gap > 0.05:  # 5% threshold
            status = "overfitting"
            explanation = f"Training accuracy ({train_acc:.4f}) significantly higher than test accuracy ({test_acc:.4f})"
            recommendations = [
                "Reduce model complexity (e.g., decrease max_depth for Random Forest)",
                "Increase regularization",
                "Collect more training data",
                "Use cross-validation for model selection"
            ]
        elif test_acc < 0.70:  # Below reasonable threshold
            status = "underfitting"
            explanation = f"Both training and test accuracy are low (test: {test_acc:.4f})"
            recommendations = [
                "Increase model complexity",
                "Add more features or feature engineering",
                "Reduce regularization",
                "Try different algorithms"
            ]
        else:
            status = "good_fit"
            explanation = f"Model shows good balance (train: {train_acc:.4f}, test: {test_acc:.4f})"
            recommendations = [
                "Model is well-balanced",
                "Consider minor hyperparameter tuning for optimization",
                "Validate on additional data if available"
            ]
        
        analysis['status'] = status
        analysis['explanation'] = explanation
        analysis['recommendations'] = recommendations
        
        log_process(f"🔍 Fitting analysis: {status.upper()}", "evaluation", explanation)
        
        for i, rec in enumerate(recommendations[:2], 1):  # Log first 2 recommendations
            log_process(f"💡 Recommendation {i}: {rec}", "info")
        
        return analysis
        
    except Exception as e:
        log_process(f"❌ Overfitting analysis failed: {str(e)}", "error")
        return None

def hyperparameter_tuning(data_prep, model_type="RandomForest"):
    """Systematic hyperparameter tuning with progress tracking"""
    log_process(f"⚙️ Starting hyperparameter tuning for {model_type}", "tuning")
    
    try:
        # Define parameter grids
        if model_type == "RandomForest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        elif model_type == "LogisticRegression":
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)
        
        log_process(f"🔧 Parameter grid defined: {len(param_grid)} parameters to tune", "tuning")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=0, return_train_score=True
        )
        
        tuning_start = time.time()
        log_process("🔄 Executing grid search cross-validation...", "tuning")
        
        grid_search.fit(data_prep['X_train_scaled'], data_prep['y_train'])
        
        tuning_time = time.time() - tuning_start
        log_process(f"✅ Hyperparameter tuning completed in {tuning_time:.2f} seconds", "success")
        
        # Extract results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        log_process(f"🏆 Best parameters found: {best_params}", "tuning", json.dumps(best_params, indent=2))
        log_process(f"🎯 Best cross-validation score: {best_score:.4f}", "tuning")
        
        # Get top 5 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_results = results_df.nlargest(5, 'mean_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'mean_train_score']
        ].to_dict('records')
        
        tuning_results = {
            'best_model': grid_search.best_estimator_,
            'best_params': best_params,
            'best_score': best_score,
            'tuning_time': tuning_time,
            'top_results': top_results,
            'all_results': grid_search.cv_results_
        }
        
        return tuning_results
        
    except Exception as e:
        log_process(f"❌ Hyperparameter tuning failed: {str(e)}", "error")
        return None

def main():
    st.title("🤖 Phase 3: Model Development & Training")
    st.markdown("**Complete Training Process with Full Accountability**")
    
    # Sidebar for process control
    st.sidebar.title("🎛️ Training Control Panel")
    st.sidebar.markdown("Monitor and control the complete training process:")
    
    # Target accuracy setting
    st.sidebar.subheader("🎯 Training Target")
    st.session_state.target_accuracy = st.sidebar.slider(
        "Target Accuracy", 0.70, 0.95, st.session_state.target_accuracy, 0.01
    )
    st.sidebar.write(f"Current target: {st.session_state.target_accuracy:.1%}")
    
    # Process status display
    st.sidebar.subheader("📋 Process Status")
    status_icons = {"pending": "⏳", "executing": "🔄", "completed": "✅", "error": "❌"}
    for step, status in st.session_state.phase3_status.items():
        st.sidebar.write(f"{status_icons[status]} {step.replace('_', ' ').title()}")
    
    # Step-by-step execution controls
    st.sidebar.subheader("🚀 Execute Training Steps")
    
    # Step 1: Data Preparation
    if st.sidebar.button("1️⃣ Prepare Training Data", key="prep_data_btn"):
        st.session_state.phase3_status['data_preparation'] = 'executing'
        prep_result = execute_with_accountability(
            "Prepare Training Data",
            prepare_training_data
        )
        if prep_result:
            st.session_state.phase3_status['data_preparation'] = 'completed'
            st.session_state.data_prep = prep_result
        else:
            st.session_state.phase3_status['data_preparation'] = 'error'
    
    # Step 2: Model Training (only if data is prepared)
    if hasattr(st.session_state, 'data_prep'):
        model_type = st.sidebar.selectbox(
            "Select Model Type:", 
            ["RandomForest", "LogisticRegression"],
            key="model_type_select"
        )
        
        if st.sidebar.button("2️⃣ Train Model", key="train_btn"):
            st.session_state.phase3_status['model_training'] = 'executing'
            training_result = execute_with_accountability(
                f"Train {model_type} Model",
                train_model,
                st.session_state.data_prep,
                model_type
            )
            if training_result:
                st.session_state.phase3_status['model_training'] = 'completed'
                st.session_state.training_result = training_result
            else:
                st.session_state.phase3_status['model_training'] = 'error'
    
    # Step 3: Model Evaluation (only if model is trained)
    if hasattr(st.session_state, 'training_result'):
        if st.sidebar.button("3️⃣ Evaluate Model", key="eval_btn"):
            st.session_state.phase3_status['model_evaluation'] = 'executing'
            eval_result = execute_with_accountability(
                "Evaluate Model Performance",
                evaluate_model,
                st.session_state.training_result,
                st.session_state.data_prep
            )
            if eval_result:
                st.session_state.phase3_status['model_evaluation'] = 'completed'
                st.session_state.evaluation_result = eval_result
            else:
                st.session_state.phase3_status['model_evaluation'] = 'error'
    
    # Step 4: Overfitting Check (only if evaluation is done)
    if hasattr(st.session_state, 'evaluation_result'):
        if st.sidebar.button("4️⃣ Check Over/Underfitting", key="fitting_btn"):
            st.session_state.phase3_status['overfitting_check'] = 'executing'
            fitting_result = execute_with_accountability(
                "Analyze Over/Underfitting",
                check_overfitting_underfitting,
                st.session_state.training_result,
                st.session_state.data_prep,
                st.session_state.evaluation_result
            )
            if fitting_result:
                st.session_state.phase3_status['overfitting_check'] = 'completed'
                st.session_state.fitting_analysis = fitting_result
            else:
                st.session_state.phase3_status['overfitting_check'] = 'error'
    
    # Step 5: Hyperparameter Tuning
    if hasattr(st.session_state, 'data_prep'):
        if st.sidebar.button("5️⃣ Tune Hyperparameters", key="tune_btn"):
            st.session_state.phase3_status['hyperparameter_tuning'] = 'executing'
            tuning_result = execute_with_accountability(
                f"Hyperparameter Tuning",
                hyperparameter_tuning,
                st.session_state.data_prep,
                getattr(st.session_state, 'training_result', {}).get('model_type', 'RandomForest')
            )
            if tuning_result:
                st.session_state.phase3_status['hyperparameter_tuning'] = 'completed'
                st.session_state.tuning_result = tuning_result
                
                # Record this iteration
                iteration = {
                    'iteration_number': len(st.session_state.training_iterations) + 1,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model_type': getattr(st.session_state, 'training_result', {}).get('model_type', 'RandomForest'),
                    'best_score': tuning_result['best_score'],
                    'best_params': tuning_result['best_params'],
                    'target_met': tuning_result['best_score'] >= st.session_state.target_accuracy
                }
                st.session_state.training_iterations.append(iteration)
                
                log_process(f"📝 Iteration {iteration['iteration_number']} recorded: Score {iteration['best_score']:.4f}", "success")
            else:
                st.session_state.phase3_status['hyperparameter_tuning'] = 'error'
    
    # Execute All Steps button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Execute Complete Training Pipeline", key="execute_all"):
        # Execute all steps in sequence
        steps = [
            ("Prepare Training Data", prepare_training_data, []),
            ("Train Model", train_model, ["data_prep", "RandomForest"]),
            ("Evaluate Model", evaluate_model, ["training_result", "data_prep"]),
            ("Analyze Fitting", check_overfitting_underfitting, ["training_result", "data_prep", "evaluation_result"]),
            ("Tune Hyperparameters", hyperparameter_tuning, ["data_prep", "RandomForest"])
        ]
        
        # Execute data preparation
        prep_result = execute_with_accountability("Prepare Training Data", prepare_training_data)
        if prep_result:
            st.session_state.data_prep = prep_result
            
            # Train model
            training_result = execute_with_accountability("Train Model", train_model, prep_result, "RandomForest")
            if training_result:
                st.session_state.training_result = training_result
                
                # Evaluate model
                eval_result = execute_with_accountability("Evaluate Model", evaluate_model, training_result, prep_result)
                if eval_result:
                    st.session_state.evaluation_result = eval_result
                    
                    # Check fitting
                    fitting_result = execute_with_accountability(
                        "Analyze Fitting", check_overfitting_underfitting, 
                        training_result, prep_result, eval_result
                    )
                    if fitting_result:
                        st.session_state.fitting_analysis = fitting_result
                    
                    # Tune hyperparameters
                    tuning_result = execute_with_accountability("Tune Hyperparameters", hyperparameter_tuning, prep_result, "RandomForest")
                    if tuning_result:
                        st.session_state.tuning_result = tuning_result
    
    # Clear log button
    if st.sidebar.button("🗑️ Clear Training Log"):
        st.session_state.phase3_log = []
        log_process("Training log cleared by user", "info")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display results based on completion status
        
        # Data preparation results
        if hasattr(st.session_state, 'data_prep'):
            st.subheader("1️⃣ Data Preparation Results")
            
            data_stats = st.session_state.data_prep['data_stats']
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Samples", data_stats['total_samples'])
                st.metric("Training Samples", data_stats['train_samples'])
            with col_b:
                st.metric("Test Samples", data_stats['test_samples'])
                st.metric("Features", data_stats['features_count'])
            with col_c:
                train_balance = data_stats['class_balance_train']
                st.metric("Train Class 0", f"{train_balance.get(0, 0):.1%}")
                st.metric("Train Class 1", f"{train_balance.get(1, 0):.1%}")
            
            # Zero handling summary
            if st.session_state.data_prep['zero_handling_log']:
                with st.expander("🔧 Zero Values Handling Details"):
                    for feature, details in st.session_state.data_prep['zero_handling_log'].items():
                        st.write(f"**{feature}:** {details['zeros_replaced']} zeros replaced with median {details['replacement_value']:.2f}")
        
        # Training results
        if hasattr(st.session_state, 'training_result'):
            st.subheader("2️⃣ Model Training Results")
            
            training = st.session_state.training_result
            
            st.markdown(f"""
            <div class="success-box">
            <h4>✅ {training['model_type']} Model Trained Successfully</h4>
            <p><strong>Training Time:</strong> {training['training_time']:.2f} seconds</p>
            <p><strong>Hyperparameters:</strong> {json.dumps(training['hyperparameters'], indent=2)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Evaluation results
        if hasattr(st.session_state, 'evaluation_result'):
            st.subheader("3️⃣ Model Evaluation Results")
            
            eval_data = st.session_state.evaluation_result
            metrics = eval_data['metrics']
            
            # Performance metrics display
            col_x, col_y = st.columns(2)
            
            with col_x:
                st.markdown("**📊 Training Set Performance:**")
                st.metric("Accuracy", f"{metrics['train']['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['train']['precision']:.4f}")
                st.metric("Recall", f"{metrics['train']['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['train']['f1_score']:.4f}")
            
            with col_y:
                st.markdown("**📊 Test Set Performance:**")
                st.metric("Accuracy", f"{metrics['test']['accuracy']:.4f}")
                st.metric("Precision", f"{metrics['test']['precision']:.4f}")
                st.metric("Recall", f"{metrics['test']['recall']:.4f}")
                st.metric("F1-Score", f"{metrics['test']['f1_score']:.4f}")
            
            # Cross-validation results
            cv_data = metrics['cross_validation']
            st.markdown(f"**🔄 Cross-Validation (5-fold):** {cv_data['mean']:.4f} ± {cv_data['std']:.4f}")
            
            # Target achievement status
            target_status = "✅ TARGET ACHIEVED" if eval_data['target_met'] else "❌ TARGET NOT MET"
            target_color = "success" if eval_data['target_met'] else "error"
            
            st.markdown(f"""
            <div class="{target_color}-box">
            <h4>{target_status}</h4>
            <p><strong>Test Accuracy:</strong> {metrics['test']['accuracy']:.4f}</p>
            <p><strong>Target:</strong> {eval_data['target_accuracy']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion matrix visualization
            cm = np.array(metrics['test']['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Test Set Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        # Overfitting analysis
        if hasattr(st.session_state, 'fitting_analysis'):
            st.subheader("4️⃣ Overfitting/Underfitting Analysis")
            
            analysis = st.session_state.fitting_analysis
            
            # Status display
            status_color = {
                'overfitting': 'warning',
                'underfitting': 'error', 
                'good_fit': 'success'
            }.get(analysis['status'], 'info')
            
            st.markdown(f"""
            <div class="{status_color}-box">
            <h4>🔍 Fitting Status: {analysis['status'].replace('_', ' ').title()}</h4>
            <p><strong>Analysis:</strong> {analysis['explanation']}</p>
            <p><strong>Training Accuracy:</strong> {analysis['train_accuracy']:.4f}</p>
            <p><strong>Test Accuracy:</strong> {analysis['test_accuracy']:.4f}</p>
            <p><strong>Gap:</strong> {analysis['accuracy_gap']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("**💡 Recommendations:**")
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.write(f"{i}. {rec}")
            
            # Learning curves
            if 'learning_curves' in analysis:
                lc = analysis['learning_curves']
                
                fig = go.Figure()
                
                # Training scores
                fig.add_trace(go.Scatter(
                    x=lc['train_sizes'],
                    y=lc['train_mean'],
                    mode='lines+markers',
                    name='Training Score',
                    line=dict(color='blue'),
                    error_y=dict(type='data', array=lc['train_std'])
                ))
                
                # Validation scores
                fig.add_trace(go.Scatter(
                    x=lc['train_sizes'],
                    y=lc['test_mean'],
                    mode='lines+markers',
                    name='Validation Score',
                    line=dict(color='red'),
                    error_y=dict(type='data', array=lc['test_std'])
                ))
                
                fig.update_layout(
                    title='Learning Curves - Training vs Validation Performance',
                    xaxis_title='Training Set Size',
                    yaxis_title='Accuracy Score',
                    hovermode='x'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Hyperparameter tuning results
        if hasattr(st.session_state, 'tuning_result'):
            st.subheader("5️⃣ Hyperparameter Tuning Results")
            
            tuning = st.session_state.tuning_result
            
            st.markdown(f"""
            <div class="success-box">
            <h4>🏆 Best Hyperparameters Found</h4>
            <p><strong>Best Cross-Validation Score:</strong> {tuning['best_score']:.4f}</p>
            <p><strong>Tuning Time:</strong> {tuning['tuning_time']:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Best parameters
            st.markdown("**⚙️ Optimal Parameters:**")
            for param, value in tuning['best_params'].items():
                st.write(f"• **{param}:** {value}")
            
            # Top results comparison
            st.markdown("**📊 Top 5 Parameter Combinations:**")
            
            top_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'CV Score': result['mean_test_score'],
                    'CV Std': result['std_test_score'],
                    'Parameters': str(result['params'])
                }
                for i, result in enumerate(tuning['top_results'])
            ])
            
            st.dataframe(top_df, use_container_width=True)
        
        # Training iterations summary
        if st.session_state.training_iterations:
            st.subheader("📊 Training Iterations Summary")
            
            iterations_df = pd.DataFrame(st.session_state.training_iterations)
            
            # Progress chart
            fig = px.line(iterations_df, x='iteration_number', y='best_score', 
                         title='Training Progress Across Iterations',
                         markers=True)
            fig.add_hline(y=st.session_state.target_accuracy, line_dash="dash", 
                         annotation_text=f"Target: {st.session_state.target_accuracy:.1%}")
            fig.update_layout(xaxis_title='Iteration', yaxis_title='Best Score')
            st.plotly_chart(fig, use_container_width=True)
            
            # Iterations table
            st.dataframe(iterations_df, use_container_width=True)
    
    with col2:
        # Live process log
        display_live_log()
        
        # Training progress summary
        st.subheader("📈 Training Progress")
        
        completed_steps = sum(1 for status in st.session_state.phase3_status.values() if status == 'completed')
        total_steps = len(st.session_state.phase3_status)
        progress = completed_steps / total_steps
        
        st.progress(progress)
        st.write(f"Progress: {completed_steps}/{total_steps} steps completed")
        
        # Current status
        if progress == 1.0:
            if hasattr(st.session_state, 'evaluation_result') and st.session_state.evaluation_result['target_met']:
                st.markdown("""
                <div class="success-box">
                <h4>🎉 Training Complete!</h4>
                <p>✅ Target accuracy achieved</p>
                <p><strong>Ready for deployment!</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                <h4>⚠️ Training Complete</h4>
                <p>❌ Target not yet achieved</p>
                <p>Consider additional tuning</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            remaining_steps = [step for step, status in st.session_state.phase3_status.items() if status == 'pending']
            if remaining_steps:
                st.markdown("**Remaining Steps:**")
                for step in remaining_steps:
                    st.write(f"• {step.replace('_', ' ').title()}")
        
        # Best model summary
        if hasattr(st.session_state, 'tuning_result'):
            st.subheader("🏆 Best Model Summary")
            
            tuning = st.session_state.tuning_result
            
            st.metric("Best Score", f"{tuning['best_score']:.4f}")
            st.metric("Target Met", "✅ Yes" if tuning['best_score'] >= st.session_state.target_accuracy else "❌ No")
            
            if hasattr(st.session_state, 'evaluation_result'):
                eval_data = st.session_state.evaluation_result
                st.metric("Test Accuracy", f"{eval_data['metrics']['test']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
