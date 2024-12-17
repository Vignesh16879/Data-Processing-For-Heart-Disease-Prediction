import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve
)


def detailed_model_analysis(y_true, y_pred, y_pred_proba = None, title_prefix = '', graph_dir = "./"):
    """
    Perform comprehensive model performance analysis
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities (optional)
    - title_prefix: Prefix for visualization titles
    """
    analysis_results = {}
    
    # 1. Classification Report
    analysis_results['classification_report'] = classification_report(
        y_true, y_pred, output_dict=True
    )
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    analysis_results['confusion_matrix'] = cm
    
    # 3. Performance Metrics Extraction
    analysis_results['metrics'] = {
        'accuracy': analysis_results['classification_report']['accuracy'],
        'macro_precision': analysis_results['classification_report']['macro avg']['precision'],
        'macro_recall': analysis_results['classification_report']['macro avg']['recall'],
        'macro_f1_score': analysis_results['classification_report']['macro avg']['f1-score']
    }
    
    # 4. Advanced ROC and Precision-Recall Curves (if probabilities provided)
    if y_pred_proba is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        analysis_results['roc'] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        analysis_results['pr_curve'] = {
            'precision': precision,
            'recall': recall
        }
    
    
    # 5. Visualization Functions
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{title} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, f'{title}_confusion_matrix.png'))
        plt.close()


    def plot_roc_curve(fpr, tpr, auc_score, title):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(graph_dir, f'{title}_roc_curve.png'))
        plt.close()


    def plot_precision_recall_curve(precision, recall, title):
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title} Precision-Recall Curve')
        plt.savefig(os.path.join(graph_dir, f'{title}_precision_recall_curve.png'))
        plt.close()
    
    
    # Generate Visualizations
    plot_confusion_matrix(cm, title_prefix)
    
    if y_pred_proba is not None:
        plot_roc_curve(
            analysis_results['roc']['fpr'], 
            analysis_results['roc']['tpr'], 
            analysis_results['roc']['auc'], 
            title_prefix
        )
        plot_precision_recall_curve(
            analysis_results['pr_curve']['precision'], 
            analysis_results['pr_curve']['recall'], 
            title_prefix
        )
    
    # 6. Error Analysis
    misclassified_indices = np.where(y_true != y_pred)[0]
    analysis_results['misclassification'] = {
        'indices': misclassified_indices,
        'count': len(misclassified_indices),
        'percentage': len(misclassified_indices) / len(y_true) * 100
    }
    
    # 7. Feature Importance (if applicable)
    # Note: This requires access to the model and feature names
    
    return analysis_results


def preprocess_data(X, categorical_features, numeric_features):
    """
    Preprocess data for model compatibility with robust handling of missing values.
    
    Parameters:
    - X: Features (DataFrame)
    - categorical_features: List of categorical column names
    - numeric_features: List of numeric column names
    
    Returns:
    - Preprocessed features as a numpy array
    - Fitted preprocessor
    """
    # Create a robust preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric features: impute missing values with median, then scale
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            
            # Categorical features: impute missing values with 'Unknown', then one-hot encode
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',  # Handle new categories in test data
                    drop='first'              # Avoid multicollinearity
                ))
            ]), categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, preprocessor


def CompareModelPerformance(models, X_test=None, y_test=None, file=None, preprocessor=None, logger: Optional[logging.Logger] = None, graph_dir = "./"):
    """
    Compare performance across multiple models with robust preprocessing.
    
    Parameters:
    - models: Dictionary of fitted models
    - X_test: Test features (optional, used if file is None)
    - y_test: Test labels
    - file: File path to the test dataset (CSV)
    - preprocessor: Fitted ColumnTransformer from training data
    - logger: Logger instance for logging messages (optional)
    """
    # Setup logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    try:
        # Load data from file if provided
        if file:
            logger.info(f"Loading test data from file: {file}")
            data = pd.read_csv(file)
            columns = data.columns.tolist()
            X_test = data.drop(columns=['target'])
            y_test = data['target']
            
            if y_test.isnull().any():
                y_test = y_test.fillna(y_test.mode()[0])

        # Validate input
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test cannot be None if no file is provided.")

        # Identify feature types dynamically
        categorical_features = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # If no preprocessor is provided, create one
        if preprocessor is None:
            logger.info("No preprocessor provided. Creating a new one.")
            _, preprocessor = preprocess_data(
                X_test, 
                categorical_features, 
                numeric_features
            )

        # Preprocess test data
        logger.info(f"Test data before preprocessing: {X_test.head()}")
        X_test_processed = preprocessor.transform(X_test)
        X_test_processed = pd.DataFrame(X_test_processed, columns=[f"feature_{i}" for i in range(X_test_processed.shape[1])])
        X_test_processed = X_test_processed.iloc[:, :22]
        new_columns = ['feature_' + str(i) for i in range(31)]
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=new_columns)
        logger.info(f"Test data after preprocessing: {X_test_processed_df.head()}")
        logger.info(f"Shape: {X_test_processed.shape}")

        # Performance comparison
        performance_comparison = {}

        for name, model in models.items():
            logger.info(f"Expected features: {model.n_features_in_}")
            logger.info(f"Actual features in preprocessed test data: {X_test_processed.shape[1]}")
            logger.info(f"Test data columns: {X_test_processed.columns}")
            logger.info(f"Test data columns (processed): {X_test_processed.columns.tolist()}")

            if model.n_features_in_ != X_test_processed.shape[1]:
                raise ValueError(f"Feature mismatch: Expected {model.n_features_in_}, got {X_test_processed.shape[1]}")

            # Check feature compatibility
            if hasattr(model, 'n_features_in_'):
                # Log and handle feature mismatch
                logger.warning(f"Model {name} expects {model.n_features_in_} features, but preprocessed data has {X_test_processed.shape[1]} features.")
                
                # Optional: Try to slice features if the preprocessed data has more features
                if X_test_processed_df.shape[1] > model.n_features_in_:
                    X_test_processed = X_test_processed.iloc[:, :model.n_features_in_]
                else:
                    raise ValueError(f"Feature mismatch for {name}: Expected {model.n_features_in_}, got {X_test_processed.shape[1]}")

            # Perform predictions
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None

            # Analyze model performance
            performance_comparison[name] = detailed_model_analysis(
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                title_prefix=name,
                graph_dir = graph_dir
            )

        return {
            "success": True,
            "message": None,
            "data": {
                "results": performance_comparison
            },
            "error": None
        }

    except Exception as e:
        # Comprehensive error handling
        error = traceback.format_exc()
        logger.error(f"[CompareModel] Error in testing process: {str(e)}")
        logger.error(f"{error}")

        return {
            "success": False,
            "message": str(e),
            "data": None,
            "error": error
        }