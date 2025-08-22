# 🍷 Wine Quality Prediction  

---

## 📜 License  
This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.  

---

## 📖 Overview  
A machine learning project that predicts the **quality of wine** based on its physicochemical properties.  
The quality scores (1–10) were converted into **three categories**:  

- **Low (1–3)**  
- **Medium (4–7)**  
- **High (8–10)**  

The project includes **data preprocessing, multiple ML models, hyperparameter tuning, model evaluation, and a web application** for predictions using **Flask**.  

---

## 🎯 Features  

- Multiclass Classification (Low, Medium, High)  
- Data Preprocessing & Feature Scaling  

### Models Implemented  
- Naive Bayes  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  

### Additional Features  
- Hyperparameter tuning using **GridSearchCV**  
- Flask Web Application for Predictions  
- Visualizations:  
  - Correlation Heatmap  
  - ROC Curves  
  - Confusion Matrix  

---

## 📁 Folder Structure  
```
wine-quality-prediction/
│
├── SQL_Data/                # Database-related files (if any)
├── static/                  # Static files for Flask (CSS, JS)
├── templates/               # HTML templates for Flask
│
├── Data.py                  # Data loading
├── EDA.py                   # Exploratory Data Analysis
├── Internship (2).ipynb     # Jupyter Notebook for experiments
├── Libraries.py             # Library imports
├── Master_Script.py         # Main pipeline script
├── Model_Fitting.py         # Model training
├── Model_Evaluation.py      # Evaluation & metrics
├── Prediction.py            # Prediction logic
├── Preprocessing.py         # Data preprocessing steps
├── Transformation.py        # Data transformation
├── app.py                   # Flask web application
├── LICENSE                  # License file
└── requirements.txt         # Required packages
```

---

## 🛠️ Setup 
### Create virtual environment and install packages  

```
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ How to Run
Jupyter Notebook (Model Training & Analysis)
```
jupyter notebook
```

Terminal
```
python app.py
```

---

## 📦 Requirements
Use requirements.txt

---

## 📊 Model Evaluation
- Confusion Matrix
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Curve

---

## ✅ Deployment
Flask for real-time prediction

---

## 🧑‍💻 Author
Developed by: Irene Betsy D

---
