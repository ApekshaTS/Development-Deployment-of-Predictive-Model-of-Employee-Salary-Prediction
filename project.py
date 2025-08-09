# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:14:35 2023

@author: 91821
"""

import pickle as pkl
import numpy as np
import streamlit as st

st.title("Employee Salary Prediction Model")
filepath="C:\\Users\\91821\\SDP79_internship\\salary_model.sav"
sal_mod= pkl.load(open(filepath, "rb"))

def pred(x):
    x=np.array(x).reshape(1,-1)
    result=sal_mod.predict(x)
    return('The salary will be: ' + str(result[0]))   
    
def main():
    years=st.number_input("Years of experience")
    new_data=years
    if st.button("Predict"):
        st.write(pred(new_data))

if __name__ == "__main__": 
    main()  
