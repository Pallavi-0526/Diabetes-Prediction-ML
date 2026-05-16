# 🩺 Diabetes Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![Internship](https://img.shields.io/badge/InternPe-Task%201-purple)

> Predicting diabetes risk from patient health data using Machine Learning — built as Task 1 of my internship at **InternPe IT Services**.

---

## 📌 Project Overview

Diabetes is one of the most common diseases affecting millions worldwide. Early detection can save lives. This project builds an end-to-end Machine Learning pipeline that predicts whether a patient is **Diabetic or Non-Diabetic** based on 8 medical features.

---

## 📊 Dataset

- **Name:** Pima Indians Diabetes Dataset
- **Samples:** 768 patients
- **Features:** 8 medical attributes
- **Target:** Outcome (0 = No Diabetes, 1 = Diabetes)

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (μU/mL) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes hereditary score |
| Age | Age in years |

---

## 🔬 ML Models Used

| Model | Accuracy | AUC Score | CV Score |
|-------|----------|-----------|----------|
| Logistic Regression | 0.688 | 0.761 | 0.754 |
| **Random Forest** ⭐ | **0.766** | **0.858** | **0.790** |
| Gradient Boosting | 0.773 | 0.856 | 0.788 |
| SVM | 0.695 | 0.796 | 0.772 |

🏆 **Best Model: Random Forest — AUC = 0.858**

---

## 📈 Visualizations

![Diabetes ML Dashboard](diabetes_charts.png)

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

**2. Install required libraries**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**3. Run the project**
```bash
python diabetes_prediction.py
```

---

## 📂 Project Structure

```
diabetes-prediction-ml/
│
├── diabetes_prediction.py   # Main ML pipeline
├── diabetes_charts.png      # Output visualizations
└── README.md                # Project documentation
```

---

## 🎯 Results

- ✅ Random Forest achieved highest AUC of **0.858**
- ✅ High-risk patient predicted as **Diabetic (85% probability)**
- ✅ Low-risk patient predicted as **Non-Diabetic (2% probability)**
- ✅ Model saved for future predictions

---

## 👩‍💻 Author

**Pallavi**
Student at Sreenivasa Institute of Technology and Management Studies
Intern at InternPe IT Services

---

## 🙏 Acknowledgements

- Thanks to **InternPe IT Services** for this learning opportunity
- Dataset: Pima Indians Diabetes Dataset
