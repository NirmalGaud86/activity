#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import pickle

# Load the Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

# App title and header
st.title('Activity Prediction App')
st.subheader('Upload a CSV file containing the features for prediction:')

# Activity name mapping
activity_mapping = {
    1: 'transient activities',
    2: 'lying',
    3: 'sitting',
    4: 'standing',
    5: 'ironing',
    6: 'vacuum cleaning',
    7: 'ascending stairs',
    8: 'descending stairs',
    9: 'walking',
    10: 'Nordic walking',
    11: 'cycling',
    12: 'running',
    13: 'rope jumping'
}

# Function to perform prediction
def predict_activity(data):
    prediction = dt_model.predict(data)
    return prediction[0]

# Upload CSV file
uploaded_file = st.file_uploader('Upload CSV', type=['csv'])

# Perform prediction if CSV file is uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write('Uploaded Data:')
    st.write(data)

    # Predict button
    if st.button('Predict'):
        prediction_result = predict_activity(data)
        st.subheader('Predicted Activity:')
        st.write(activity_mapping[prediction_result])

