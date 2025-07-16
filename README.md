
# 📉 Customer Churn Prediction

This project predicts whether a customer will churn (leave a service) based on various features such as tenure, service type, and payment method. It uses data preprocessing, class balancing, and machine learning models to build a reliable churn prediction system.

---

## 🚀 Key Features

- Data preprocessing and cleaning
- Label encoding of categorical variables
- Handling class imbalance using **SMOTE**
- Model training using:
  - Decision Tree
  - Random Forest
  - XGBoost
- Cross-validation to compare models
- Saving trained model and encoders with `pickle`
- Making predictions on new input data

---

## 📁 Dataset

Dataset: `data.csv`  
It contains telecom customer data with features like:

- `gender`, `SeniorCitizen`, `Partner`, `tenure`
- `PhoneService`, `InternetService`, `Contract`
- `MonthlyCharges`, `TotalCharges`
- `Churn` (target column)

---

## 🧪 Steps Performed

### 1. **Import Libraries**
Used libraries like `pandas`, `scikit-learn`, `seaborn`, `xgboost`, `imblearn`, and `pickle`.

### 2. **Load & Explore Data**
- Read data from CSV
- Checked shape, info, and missing values
- Dropped irrelevant column: `customerID`

### 3. **Data Cleaning**
- Replaced blank `TotalCharges` with `0.0` and converted to float
- Handled categorical columns using `LabelEncoder`

### 4. **Exploratory Data Analysis**
- Histograms and boxplots for numeric features
- Count plots for categorical features
- Correlation heatmap

### 5. **Label Encoding**
- All categorical columns were encoded
- Saved all encoders to `encoders.pkl` for later use

### 6. **Train-Test Split**
Split data into `80%` training and `20%` testing.

### 7. **Class Balancing**
Used `SMOTE` to oversample the minority class (Churn = 1).

### 8. **Model Training**
Trained 3 models:
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `XGBClassifier`

Each model was evaluated using **5-fold cross-validation**.

### 9. **Evaluation**
The best model (`RandomForestClassifier`) was evaluated on test data using:
- Accuracy
- Confusion Matrix
- Classification Report

### 10. **Model Saving**
- Saved the final model to `customer_churn_model.pkl`
- Also saved the list of feature names

---

## 🔮 Make Predictions on New Data

- Load model and encoders
- Prepare new input as a dictionary
- Apply saved encoders
- Predict churn or not
- Output prediction and probability

---

## 🧾 Example Input for Prediction

```python
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}
````

---

## 📦 Files Included

* `data.csv` – Dataset
* `Customer Churn Prediction.ipynb` – Main notebook
* `encoders.pkl` – Saved encoders for label encoding
* `customer_churn_model.pkl` – Final trained model with feature names

---

## 📊 Evaluation Metrics Used

* **Accuracy**
* **Confusion Matrix**
* **Precision, Recall, F1-Score**

---

## ✅ Requirements

Install required libraries using:

```bash
pip install pandas scikit-learn seaborn matplotlib xgboost imbalanced-learn
```

---

## 🧠 Future Improvements

* Use OneHotEncoding for better handling of categorical variables
* Add a web interface using Streamlit or Flask
* Include feature importance visualization
* Deploy as an API or cloud service

---

## 👨‍💻 Author

M. Zohaib Shahid
Data Science Enthusiast

---
