import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn


encoder_Application_mode = joblib.load("model/encoder_Application_mode.joblib")
encoder_Attendance_time = joblib.load("model/encoder_Attendance_time.joblib")
encoder_Course = joblib.load("model/encoder_Course.joblib")
encoder_Debtor = joblib.load("model/encoder_Debtor.joblib")
encoder_Displaced = joblib.load("model/encoder_Displaced.joblib")
encoder_Educational_special_needs = joblib.load("model/encoder_Educational_special_needs.joblib")
encoder_Fathers_occupation = joblib.load("model/encoder_Fathers_occupation.joblib")
encoder_Fathers_qualification = joblib.load("model/encoder_Fathers_qualification.joblib")
encoder_Gender = joblib.load("model/encoder_Gender.joblib")
encoder_International = joblib.load("model/encoder_International.joblib")
encoder_Marital_status = joblib.load("model/encoder_Marital_status.joblib")
encoder_Mothers_occupation = joblib.load("model/encoder_Mothers_occupation.joblib")
encoder_Mothers_qualification = joblib.load("model/encoder_Mothers_qualification.joblib")
encoder_Nacionality = joblib.load("model/encoder_Nacionality.joblib")
encoder_Previous_qualification = joblib.load("model/encoder_Previous_qualification.joblib")
encoder_Scholarship_holder = joblib.load("model/encoder_Scholarship_holder.joblib")
encoder_target = joblib.load("model/encoder_target.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("model/encoder_Tuition_fees_up_to_date.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Application_order = joblib.load("model/scaler_Application_order.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_(approved).joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("model/scaler_Curricular_units_1st_sem_(credited).joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_(enrolled).joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_(evaluations).joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_(grade).joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_(without_evaluations).joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_(approved).joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("model/scaler_Curricular_units_2nd_sem_(credited).joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_(enrolled).joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_(evaluations).joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_(grade).joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_(without_evaluations).joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_(grade).joblib")



def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    df['Marital_status'] = encoder_Marital_status.transform(data['Marital_status'])[0]
    df['Application_mode'] = encoder_Application_mode.transform(data['Application_mode'])[0]
    df['Application_order'] = scaler_Application_order.transform(np.asarray(data['Application_order']).reshape(-1,1))[0]
    df['Course'] = encoder_Course.transform(data['Course'])[0]
    df['Attendance_time'] = encoder_Attendance_time.transform(data['Attendance_time'])[0]
    df['Previous_qualification'] = encoder_Previous_qualification.transform(data['Previous_qualification'])[0]
    df['Previous_qualification_(grade)'] = scaler_Previous_qualification_grade.transform(np.asarray(data['Previous_qualification_(grade)']).reshape(-1,1))[0]
    df['Nacionality'] = encoder_Nacionality.transform(data['Nacionality'])[0]
    df["Mother's_qualification"] = encoder_Mothers_qualification.transform(data["Mother's_qualification"])[0]
    df["Father's_qualification"] = encoder_Fathers_qualification.transform(data["Father's_qualification"])[0]
    df["Mother's_occupation"] = encoder_Mothers_occupation.transform(data["Mother's_occupation"])[0]
    df["Father's_occupation"] = encoder_Fathers_occupation.transform(data["Father's_occupation"])[0]
    df['Displaced'] = encoder_Displaced.transform(data['Displaced'])[0]
    df['Educational_special_needs'] = encoder_Educational_special_needs.transform(data['Educational_special_needs'])[0]
    df['Debtor'] = encoder_Debtor.transform(data['Debtor'])[0]
    df['Tuition_fees_up_to_date'] = encoder_Tuition_fees_up_to_date.transform(data['Tuition_fees_up_to_date'])[0]
    df['Gender'] = encoder_Gender.transform(data['Gender'])[0]
    df['Scholarship_holder'] = encoder_Scholarship_holder.transform(data['Scholarship_holder'])[0]
    df['Admission_grade'] = scaler_Admission_grade.transform(np.asarray(data['Admission_grade']).reshape(-1,1))[0]
    df['International'] = encoder_International.transform(data['International'])[0]
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))[0]
    df['Curricular_units_1st_sem_(credited)'] = scaler_Curricular_units_1st_sem_credited.transform(np.asarray(data['Curricular_units_1st_sem_(credited)']).reshape(-1,1))[0]
    df['Curricular_units_1st_sem_(enrolled)'] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data['Curricular_units_1st_sem_(enrolled)']).reshape(-1,1))[0]
    df['Curricular_units_1st_sem_(evaluations)'] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data['Curricular_units_1st_sem_(evaluations)']).reshape(-1,1))[0]
    df['Curricular_units_1st_sem_(approved)'] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data['Curricular_units_1st_sem_(approved)']).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_(grade)"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_(grade)"]).reshape(-1,1))[0]
    df["Curricular_units_1st_sem_(without_evaluations)"] = scaler_Curricular_units_1st_sem_without_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_(without_evaluations)"]).reshape(-1,1))[0]
    df['Curricular_units_2nd_sem_(credited)'] = scaler_Curricular_units_2nd_sem_credited.transform(np.asarray(data['Curricular_units_2nd_sem_(credited)']).reshape(-1,1))[0]
    df['Curricular_units_2nd_sem_(enrolled)'] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data['Curricular_units_2nd_sem_(enrolled)']).reshape(-1,1))[0]
    df['Curricular_units_2nd_sem_(evaluations)'] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data['Curricular_units_2nd_sem_(evaluations)']).reshape(-1,1))[0]
    df['Curricular_units_2nd_sem_(approved)'] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data['Curricular_units_2nd_sem_(approved)']).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_(grade)"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_(grade)"]).reshape(-1,1))[0]
    df["Curricular_units_2nd_sem_(without_evaluations)"] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_(without_evaluations)"]).reshape(-1,1))[0]
   
    return df

model = joblib.load("model/rdf_model.joblib")
result_target = joblib.load("model/encoder_target.joblib")

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result



st.header('Credit Scoring App (Prototype)')

data = pd.DataFrame()
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    Marital_status = st.selectbox(label='Marital_status', options=encoder_Marital_status.classes_, index=1)
    data["Marital_status"] = [Marital_status]
 
with col2:
    Application_mode = st.selectbox(label='Application_mode', options=encoder_Application_mode.classes_, index=1)
    data["Application_mode"] = [Application_mode]
 
with col3:
    Course = st.selectbox(label='Course', options=encoder_Course.classes_, index=5)
    data["Course"] = Course

with col4:
    Attendance_time = st.selectbox(label='Attendance_time', options=encoder_Attendance_time.classes_, index=1)
    data["Attendance_time"] = Attendance_time 

col1, col2, col3, col4 = st.columns(4)
 
with col1:
    Previous_qualification = st.selectbox(label='Previous_qualification', options=encoder_Previous_qualification.classes_, index=1)
    data["Previous_qualification"] = [Previous_qualification]
 
with col2:
    Nacionality = st.selectbox(label='Nacionality', options=encoder_Nacionality.classes_, index=1)
    data["Nacionality"] = [Nacionality]
 
with col3:
    Displaced = st.selectbox(label='Displaced', options=encoder_Displaced.classes_, index=1)
    data["Displaced"] = Displaced

with col4:
    Educational_special_needs = st.selectbox(label='Educational_special_needs', options=encoder_Educational_special_needs.classes_, index=1)
    data["Educational_special_needs"] = Educational_special_needs

col1, col2, col3, col4 = st.columns(4)
 
with col1:
    Mothers_qualification = st.selectbox(label="Mother's_qualification", options=encoder_Mothers_qualification.classes_, index=1)
    data["Mother's_qualification"] = [Mothers_qualification]

with col2:
    Mothers_occupation = st.selectbox(label="Mother's_occupation", options=encoder_Mothers_occupation.classes_, index=1)
    data["Mother's_occupation"] = [Mothers_occupation]
 
with col3:
    Fathers_qualification = st.selectbox(label="Father's_qualification", options=encoder_Fathers_qualification.classes_, index=1)
    data["Father's_qualification"] = Fathers_qualification

with col4:
    Fathers_occupation = st.selectbox(label="Father's_occupation", options=encoder_Fathers_occupation.classes_, index=1)
    data["Father's_occupation"] = Fathers_occupation

col1, col2, col3 = st.columns(3)
 
with col1:
    Debtor = st.selectbox(label='Debtor', options=encoder_Debtor.classes_, index=1)
    data["Debtor"] = [Debtor]

with col2:
    Tuition_fees_up_to_date = st.selectbox(label='Tuition_fees_up_to_date', options=encoder_Tuition_fees_up_to_date.classes_, index=1)
    data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]
 
with col3:
    Gender = st.selectbox(label='Gender', options=encoder_Gender.classes_, index=1)
    data["Gender"] = Gender

col1, col2 = st.columns(2)
 
with col1:
    Scholarship_holder = st.selectbox(label='Scholarship_holder', options=encoder_Scholarship_holder.classes_, index=1)
    data["Scholarship_holder"] = [Scholarship_holder]

with col2:
    International = st.selectbox(label='International', options=encoder_International.classes_, index=1)
    data["International"] = [International]

col1, col2, col3, col4 = st.columns(4)
 
with col1:
       
    Application_order = int(st.number_input(label='Application_order', value=4))
    data["Application_order"] = Application_order
 
with col2:
    Previous_qualification_grade = int(st.number_input(label='Previous_qualification_(grade)', value=100))
    data["Previous_qualification_(grade)"] = Previous_qualification_grade
 
with col3:
    Admission_grade = int(st.number_input(label='Admission_grade', value=100))
    data["Admission_grade"] = Admission_grade
 
with col4:
    Age_at_enrollment = float(st.number_input(label='Age_at_enrollment', value=17))
    data["Age_at_enrollment"] = Age_at_enrollment

col1, col2, col3, col4 = st.columns(4)
 
with col1:
    Curricular_units_1st_sem_credited= int(st.number_input(label='Curricular_units_1st_sem_(credited)', value=0.541817))
    data["Curricular_units_1st_sem_(credited)"] = Curricular_units_1st_sem_credited
 
with col2:
    Curricular_units_1st_sem_enrolled= int(st.number_input(label='Curricular_units_1st_sem_(enrolled)', value=6.232143))
    data["Curricular_units_1st_sem_(enrolled)"] = Curricular_units_1st_sem_enrolled
 
with col3:
    Curricular_units_1st_sem_evaluations = int(st.number_input(label='Curricular_units_1st_sem_(evaluations)', value=8.063291))
    data["Curricular_units_1st_sem_(evaluations)"] = Curricular_units_1st_sem_evaluations
 
with col4:
    Curricular_units_1st_sem_approved = int(st.number_input(label='Curricular_units_1st_sem_(approved)', value=4.435805))
    data["Curricular_units_1st_sem_(approved)"] = Curricular_units_1st_sem_approved
 
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:  
    Curricular_units_2nd_sem_credited= int(st.number_input(label='Curricular_units_2nd_sem_(credited)', value=0.541817))
    data["Curricular_units_2nd_sem_(credited)"] = Curricular_units_2nd_sem_credited
 
with col2:
    Curricular_units_2nd_sem_enrolled= int(st.number_input(label='Curricular_units_2nd_sem_(enrolled)', value=6.232143))
    data["Curricular_units_2nd_sem_(enrolled)"] = Curricular_units_2nd_sem_enrolled
 
with col3:
    Curricular_units_2nd_sem_evaluations = int(st.number_input(label='Curricular_units_2nd_sem_(evaluations)', value=8.063291))
    data["Curricular_units_2nd_sem_(evaluations)"] = Curricular_units_2nd_sem_evaluations
 
with col4:
    Curricular_units_2nd_sem_approved = int(st.number_input(label='Curricular_units_2nd_sem_(approved)', value=4.435805))
    data["Curricular_units_2nd_sem_(approved)"] = Curricular_units_2nd_sem_approved
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    Curricular_units_1st_sem_grade= int(st.number_input(label='Curricular_units_1st_sem_(grade)', value=4))
    data["Curricular_units_1st_sem_(grade)"] = Curricular_units_1st_sem_grade
 
with col2:
    Curricular_units_1st_sem_without_evaluations= int(st.number_input(label='Curricular units 1st sem (without_evaluations)', value=0.15))
    data["Curricular_units_1st_sem_(without_evaluations)"] = Curricular_units_1st_sem_without_evaluations
 
with col3:
    Curricular_units_2nd_sem_grade= int(st.number_input(label='Curricular_units_2nd_sem_(grade)', value=4))
    data["Curricular_units_2nd_sem_(grade)"] = Curricular_units_2nd_sem_grade
 
with col4:
    Curricular_units_2nd_sem_without_evaluations= int(st.number_input(label='Curricular units 2nd sem (without_evaluations)', value=0.15))
    data["Curricular_units_2nd_sem_(without_evaluations)"] = Curricular_units_2nd_sem_without_evaluations

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Credit Scoring: {}".format(prediction(new_data)))