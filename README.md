Customer Churn Prediction Using Machine Learning

This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to leave a service, enabling companies to take proactive retention actions.
🚀 Project Overview

Customer churn is a critical challenge for businesses, especially in competitive markets. Using the dataset collected from Kaggle, this project applies data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning techniques to build an effective churn prediction model.

📂 Project Structure

📁 Customer-Churn-Prediction
│── 📄 README.md
│── 📓 Customer_Churn_Prediction.ipynb
│── 📁 data/
│     └── customer_churn_data.csv
│── 📁 models/
│     └── saved_models
│── 📁 images/
│     └── visualizations, charts

🧠 Key Features of the Project
✔️ Data Preprocessing

Handling missing values

Encoding categorical features

Feature scaling

Removing multicollinearity

✔️ Exploratory Data Analysis

Churn distribution

Correlation heatmap

Demographic and service usage patterns

Identifying key factors influencing churn

✔️ Machine Learning Models Used

Logistic Regression

Random Forest Classifier

XGBoost (Optional)

Support Vector Machine

✔️ Model Evaluation

Accuracy Score

Precision, Recall, F1-Score

Confusion Matrix

ROC–AUC Curve

✔️ Best Model Performance

Logistic Regression achieved 95% accuracy (based on your project)

🛠️ Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Notebook

📊 Results & Insights

Customers using month-to-month contracts showed higher churn.

High charges and lower tenure correlated strongly with churn.

Logistic Regression provided the best performance for your data.

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/Suyog308/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Jupyter Notebook
jupyter notebook Customer_Churn_Prediction.ipynb
