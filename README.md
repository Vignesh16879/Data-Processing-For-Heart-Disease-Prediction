# **Data Processing For Heart Disease Prediction**


### **Overview**  
This project predicts the presence of heart disease using machine learning models. The dataset contains 13 input features and 1 binary target variable (0: No Heart Disease, 1: Heart Disease). The models developed include Logistic Regression, Random Forest, and XGBoost, with preprocessing steps ensuring data quality and improved model accuracy.

---

## **Project Structure**  

```
Heart_Disease_Prediction/
│
├── main.ipynb              # Main Jupyter Notebook for the analysis pipeline
│
├── src/                    
│   ├── __init__.py         # Package initialization file
│   ├── preprocessing.py    # Custom functions for data preprocessing
│   ├── models.py           # Model training, testing, and evaluation functions
│   └── utils.py            # Utility functions for visualization and metrics
│
├── src/Data/Raw/           # Raw dataset files
│   └── heart_disease.csv   # Heart Disease dataset
│
├── results/                # Model outputs, plots, and metrics
│
└── README.md               # Project description and setup instructions
```

---

## **Installation Instructions**  

1. Clone the repository:  
   ```bash
   git clone <repository-url>
   cd Heart_Disease_Prediction
   ```

2. Set up a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install required libraries:  
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**  

1. Open the Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
2. Run `main.ipynb` to execute the analysis pipeline, including:  
   - Data preprocessing (handling missing values, encoding, scaling)  
   - Exploratory Data Analysis (EDA)  
   - Hypothesis testing  
   - Model training and evaluation  

---

## **Custom Functions**  

- **preprocessing.py**  
   - Handles missing values  
   - Categorical variable encoding  
   - Feature scaling and balancing  
   - Outlier detection and removal  

- **models.py**  
   - Train and evaluate Logistic Regression, Random Forest, and XGBoost models  
   - Compare performance between scaled and unscaled datasets  

- **utils.py**  
   - Utility functions for generating plots (e.g., correlation matrix, distributions)  
   - Model evaluation metrics like precision, recall, and F1-score  

---

## **Results**  

The XGBoost model achieved the best performance:  
- **Accuracy:** 89%  
- **Precision:** 88%  
- **Recall:** 87%  

**Comparison Between Full Dataset and Scaled Dataset:**  
- Scaling reduced training time by ~40%, with a minor accuracy drop of ~2%.  

---

## **Future Improvements**  

1. Implement advanced feature engineering to improve prediction accuracy.  
2. Explore deep learning models for capturing complex patterns.  
3. Expand the dataset to increase generalizability across diverse populations.  

---

## **References**  

1. Dataset Source: [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
2. SMOTE Implementation: Chawla et al., SMOTE (2002).  
3. Scikit-learn Documentation: [Link](https://scikit-learn.org)  

---

**Contributors:**  
- **Vignesh Goswami (#2020152)**: Preprocessing, hypothesis testing, model evaluation.  
- **Anmol Kaw (#2021234)**: EDA, model development, results comparison.  

---  
