import os
import pickle
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__), 'soring_pkl')

with open(model_path, 'rb') as file:
    soring = pickle.load(file)

st.title('💳Credit_Scoring_Mobel🪪')
st.write('Advice To Check Your Credit Scoring Fast')

countryCode = st.number_input("countyCode:", min_value=1, max_value=2468, value=391)
customerID = st.number_input("customerID:", min_value=1, max_value=2468, value=25)
invoiceNumber = st.number_input("invoiceNumber:", min_value=1, max_value=2468, value=25)
Disputed = st.selectbox("Disputed:", ['Yes', 'No'])
PaperlessBill = st.selectbox("PaperlessBill:", ['Paper', 'Electronic'])

if st.button("predict DaysLate"):
    disputed_value = 1 if Disputed == 'Yes' else 0
    paperless_value = 1 if PaperlessBill == 'Paper' else 0

    # The trained model expects 11 feature columns; fill the missing columns with 0
    input_data = [[
        countryCode,
        customerID,
        0,
        invoiceNumber,
        0,
        0,
        0,
        disputed_value,
        0,
        paperless_value,
        0
    ]]

    prediction = soring.predict(input_data)
    st.success(f"Prediction done Successfully ! Result:{prediction[0]}")
else:
    st.info('Prdiction button per click karo')

