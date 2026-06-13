import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def preprocess(df):
    df = df.copy()

    # 1. Drop Patient_ID
    if 'Patient_ID' in df.columns: 
        df.drop(columns=['Patient_ID'], inplace=True)

    # 2. Split Blood_Pressure → Systolic + Diastolic
    if 'Blood_Pressure' in df.columns:
        bp = df['Blood_Pressure'].str.split('/', expand=True).astype(int)
        df['Systolic_BP']  = bp[0]
        df['Diastolic_BP'] = bp[1]
        df.drop(columns=['Blood_Pressure'], inplace=True)

    # 3. Encode Gender  (Male=0, Female=1)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # 4. Encode Diagnosis_ICD as numeric code
    if 'Diagnosis_ICD' in df.columns:
        df['Diagnosis_Code'] = df['Diagnosis_ICD'].astype('category').cat.codes
        df.drop(columns=['Diagnosis_ICD'], inplace=True)

    # 5. Create Risk Label (Target Variable)
    #    High Risk = 1,  Low Risk = 0
    df['Risk_Label'] = (
        (df['Age']                 > 60)  |
        (df['Systolic_BP']         > 140) |
        (df['Blood_Glucose_mgdL']  > 200) |
        (df['SpO2_%']              < 92)  |
        (df['Creatinine_mgdL']     > 2.0) |
        (df['Previous_Admissions'] > 3)
    ).astype(int)

    # 6. Feature Engineering
    df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']

    df['BMI_Category'] = pd.cut(
        df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
        labels=[0, 1, 2, 3]).astype(int)

    df['Age_Group'] = pd.cut(
        df['Age'], bins=[0, 30, 50, 65, 100],
        labels=[0, 1, 2, 3]).astype(int)

    df['Glucose_Risk'] = (df['Blood_Glucose_mgdL'] > 126).astype(int)
    df['Kidney_Risk']  = (df['Creatinine_mgdL']    > 1.5).astype(int)
    df['LowOxygen']    = (df['SpO2_%']             < 94).astype(int)
    df['HighBUN']      = (df['BUN_mgdL']           > 25).astype(int)

    # 7. Fill any nulls with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"✅ Preprocessed: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Risk Label → {df['Risk_Label'].value_counts().to_dict()}")
    return df