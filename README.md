# **Data Processing For Heart Disease Prediction**


### **1. Overview of the Dataset and Problem Statement**
The dataset consists of **13 input features** and **1 target variable** aimed at predicting heart disease presence. The target variable is binary:
- **0**: Healthy (No Heart Disease)
- **1**: Heart Disease

**Key Features:**
- **Patient Demographics:** Age, Sex
- **Clinical Measurements:** Resting blood pressure (trestbps), Cholesterol (chol)
- **Medical Tests:** Chest Pain Type (cp), Fasting Blood Sugar (fbs), Resting ECG (restecg), Maximum Heart Rate (thalach), Exercise-Induced Angina (exang), ST depression (oldpeak)

**Problem Statement:**
Using this medical dataset, we aim to develop a data processing pipeline and machine learning models to accurately predict the presence of heart disease, enabling early detection and improved patient outcomes.

---

### **2. Project Folder Structure**
The project folder is organized as follows:

```
Heart_Disease_Prediction/
│   ├── README.md              # Project description and setup instructions
│   ├── logs.txt               # Log file to track processes
│   ├── main.ipynb             # Main Jupyter Notebook for the analysis pipeline
│   └── src/                   
│       ├── Data/              # Data directory
│       │   ├── Cleaned/       # Cleaned dataset files
│       │   │   └── cleaned.csv
│       │   ├── Merged/        # Merged dataset files
│       │   │   └── merged.csv
│       │   ├── Processed/     # Processed data files
│       │   │   ├── balanced_data.csv
│       │   │   ├── engineered_data.csv
│       │   │   ├── handlemissingvalues.csv
│       │   │   ├── outliers_removed.csv
│       │   │   ├── processed.csv
│       │   │   ├── processedencodeddata.csv
│       │   │   ├── scaled_data.csv
│       │   │   └── selected_features.csv
│       │   ├── Raw/           # Raw datasets
│       │   │   ├── heart_01.csv
│       │   │   ├── heart_02.csv
│       │   │   ├── heart_03.csv
│       │   │   └── heart_04.csv
│       │   ├── Scaled/        # Scaled dataset files
│       │   │   └── data.csv
│       │   ├── Testing/       # Testing data
│       │   │   └── data.csv
│       │   └── Training/      # Training data
│       │       └── data.csv
│       ├── DataPreProcessing/ # Data preprocessing scripts
│       │   ├── BalancingDataset.py
│       │   ├── DataCleaning.py
│       │   ├── DataEncodingCategoricalVariables.py
│       │   ├── DataFeatureEngineering.py
│       │   ├── DataFeatureScaling.py
│       │   ├── DataFeatureSelection.py
│       │   ├── DataHandleMissingValues.py
│       │   └── DataHandlingOutlier.py
│       ├── EDA/               # Exploratory Data Analysis
│       │   └── eda.py
│       ├── Graphs/            # Graphs and visualizations
│       │   ├── EDA/           # EDA graphs in PNG format
│       │   └── ModelsAnalysis/ # ML Models analysis graphs in PNG format
│       ├── Helper/            # Helper scripts
│       │   ├── dataloader.py
│       │   ├── datasaver.py
│       │   ├── filters.py
│       │   └── merger.py
│       ├── HypothesisTesting/ # Hypothesis testing scripts
│       │   ├── HypothesisTests.py
│       │   └── ValidateTests.py
│       └── Training/          # Training scripts for ML models
│           ├── compare.py
│           ├── helper.py
│           ├── scale.py
│           ├── test.py
│           └── train.py
```

---

### **3. Challenges Faced and Solutions**

1. **Missing Values:**
   - **Challenge:** Features like `ca` and `thal` contained missing values.
   - **Solution:** Imputed missing numerical values using the median and categorical values using the mode to preserve data consistency.

2. **Categorical Data Encoding:**
   - **Challenge:** Features such as `sex` and `cp` were categorical and required numerical transformation.
   - **Solution:** Binary encoding for `sex` and one-hot encoding for nominal features like `cp`.

3. **Imbalanced Dataset:**
   - **Challenge:** Heart disease cases (target = 1) were underrepresented.
   - **Solution:** Applied SMOTE (Synthetic Minority Over-Sampling Technique) to balance the dataset, ensuring equal representation of both classes.

4. **Feature Scaling:**
   - **Challenge:** Features had varying magnitudes, which could affect distance-based models.
   - **Solution:** Used standardization (z-score normalization) to ensure equal contribution of features.

5. **Outlier Detection:**
   - **Challenge:** Extreme values in features like `chol` and `trestbps` could bias the model.
   - **Solution:** Removed outliers using the interquartile range (IQR) method.

---

### **4. Group Contributions**
- **Vignesh Goswami:**
   - Data preprocessing (handling missing values, encoding categorical variables, feature scaling).
   - Hypothesis testing (chi-square and t-tests).
   - Model evaluation and performance analysis.
   - Dataset exploration and visualization.
   - Implementation of machine learning models (Logistic Regression, Random Forest, XGBoost).
   - Results comparison (scaled vs. unscaled datasets).
- **Anmol Kaw:**
   - Data Collection.

---

### **5. References**
1. **Dataset Source:** UCI Machine Learning Repository: Heart Disease Dataset.
2. **SMOTE Technique:** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
3. **Chi-Square Test Reference:** Statistics and Probability Tutorials, Khan Academy.
4. **Machine Learning Models:** Pedregosa et al., Scikit-learn: Machine Learning in Python.
