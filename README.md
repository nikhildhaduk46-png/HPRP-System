# 🏥 Hospital Patient Risk Prediction System (HPRP)

## 📌 Project Overview

The **Hospital Patient Risk Prediction System (HPRP)** is a machine learning-based healthcare application designed to predict the risk level of patients using medical and demographic information. The system assists healthcare professionals in identifying high-risk patients at an early stage, enabling timely treatment and improved clinical decision-making.

Built with **Python** and **Streamlit**, the application provides an interactive and user-friendly web interface that allows users to enter patient information and receive instant risk predictions. By leveraging machine learning algorithms, the system helps improve patient outcomes, optimize healthcare resources, and support data-driven healthcare management.

---

## 🚀 Features

* Predicts patient risk levels using machine learning models.
* User-friendly web interface built with Streamlit.
* Supports real-time patient risk assessment.
* Data preprocessing and cleaning pipeline.
* Model training and evaluation using multiple machine learning algorithms.
* Instant prediction results with a single click.
* Visualization and reporting capabilities.
* Helps healthcare professionals make informed decisions.

---

## 🛠️ Technology Stack

### Front End

* Streamlit

### Back End

* Python

### Libraries Used

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

### Extra Libraries

* ReportLab
* Joblib

---

## 📊 Dataset

The project uses a dataset containing approximately **10,000 patient records** stored in CSV format. The dataset includes important healthcare-related information such as:

* Age
* Gender
* Medical History
* Vital Signs
* Health Indicators
* Risk Factors

The data is preprocessed before training to ensure better model performance and prediction accuracy.

---

## ⚙️ Project Workflow

1. Load patient data from CSV files.
2. Perform data cleaning and preprocessing.
3. Handle missing values and data transformation.
4. Train machine learning models on processed data.
5. Evaluate model performance using accuracy and other metrics.
6. Select the best-performing model.
7. Save the trained model using Joblib.
8. Deploy the prediction system using Streamlit.
9. Allow users to enter patient details and obtain risk predictions instantly.

---

## 🤖 Machine Learning Process

The system follows a standard machine learning pipeline:

* Data Collection
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Model Selection
* Prediction Deployment

Various machine learning algorithms can be tested and compared to identify the most accurate model for patient risk prediction.

---

## 📁 Project Structure

```bash
Hospital-Patient-Risk-Prediction-System/
│
├── app.py
├── model.pkl
├── dataset.csv
├── requirements.txt
├── report.pdf
├── assets/
│
├── notebooks/
│   └── model_training.ipynb
│
└── README.md
```

---

## ▶️ Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/Hospital-Patient-Risk-Prediction-System.git
cd Hospital-Patient-Risk-Prediction-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🎯 Objectives

* Predict patient health risks accurately.
* Support healthcare professionals in decision-making.
* Enable early detection of potential health complications.
* Improve patient care and treatment planning.
* Reduce hospital readmission rates.
* Optimize healthcare resource utilization.

---

## 📈 Future Enhancements

* Integration with Electronic Health Records (EHR).
* Real-time patient monitoring.
* Deep Learning-based prediction models.
* Cloud deployment support.
* Advanced visualization dashboards.
* Multi-disease risk prediction capabilities.

---

## 👨‍💻 Author

**Nikhil Dhaduk**

---

## 📜 License

This project is developed for educational and research purposes. Feel free to modify and enhance it according to your requirements.
