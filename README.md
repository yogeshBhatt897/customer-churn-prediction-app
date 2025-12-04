#  Customer Churn Prediction App

A machine learning project that predicts whether a telecom customer will churn based on their service usage, contract type, and billing information.

This project includes:

- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Logistic Regression churn prediction model  
- Saved model (`churn_model.pkl`)  
- Interactive Streamlit web app  
- Full Jupyter Notebook analysis  


---

## Project Overview

This project uses the **Telco Customer Churn dataset** to predict whether a customer will leave a telecom service. The goal is to help businesses identify at-risk customers and take action to reduce churn.

The project demonstrates:

- Data analysis and visualization  
- Feature engineering  
- Binary classification model  
- Model evaluation  
- Deployment in a Streamlit web app  

---

## Data Cleaning Steps

The following preprocessing steps were applied:

- Dropped `customerID` (non-predictive)  
- Converted `TotalCharges` to numeric  
- Removed rows with missing `TotalCharges`  
- Replaced `"No internet service"` → `"No"`  
- Replaced `"No phone service"` → `"No"`  
- Converted `Churn` to numeric (`Yes = 1`, `No = 0`)  
- One-hot encoded categorical columns  
- Scaled numerical features with `StandardScaler`  

---

## Exploratory Data Analysis (EDA)

The notebook includes multiple visualizations such as:

- Churn Distribution  
- Churn by Contract Type  
- Monthly Charges vs Churn  
- Tenure Distribution  
- Churn by Payment Method  
- Correlation Heatmap  

### Key Insights

- **Month-to-month** contract customers churn the most  
- **Higher monthly charges** correlate with churn  
- **Low-tenure customers** churn more frequently  
- **Electronic Check** has the highest churn rate  

---

## Machine Learning Model

A **Logistic Regression** model was used due to its simplicity, interpretability, and effectiveness in binary classification.

### Model Performance

- **Accuracy:** ~80%  
- Strong performance on majority (non-churn) class  
- Moderate recall for churn class  
- Feature importance provides business insights  

Model evaluation includes:

- Confusion Matrix  
- Classification Report  
- Feature Importance table  

---

## Streamlit Web App

The Streamlit app makes churn prediction interactive.

### Run the app:

```bash
streamlit run app.py
```
---

### The app:

- Takes user input for customer attributes  
- Encodes and scales the input  
- Loads the saved machine learning model  
- Outputs the predicted churn status and probability  

---

## Dataset Source

**Telco Customer Churn Dataset (Kaggle):**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🔧 Installation

### Clone the repository:

```bash
git clone https://github.com/yogeshBhatt897/customer-churn-prediction-app.git
cd customer-churn-prediction-app
```
### Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Run the Notebook:

Open **Final Project.ipynb** in Jupyter Notebook or VS Code.

---

## Future Improvements

- Experiment with Random Forest, XGBoost, or Gradient Boosting  
- Balance classes using SMOTE to improve churn recall  
- Create a Power BI or Tableau dashboard  
- Deploy the Streamlit app to the cloud (Render, Heroku, Azure, etc.)  
- Add a REST API endpoint for real-time predictions  

