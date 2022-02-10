# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:59:38 2021

@author: VRUTIKA
"""

import warnings
warnings.filterwarnings('ignore')

# imports
import pandas as pd

# import streamlit
import streamlit as st

import pickle

filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

filename = 'vars.pkl'
fe = pickle.load(open(filename, 'rb'))


def get_category(article):
    
    tf = fe.transform([article])
    category = model.predict(tf)
    
    return category
    
def get_collective(csv):
    
    tf = fe.transform(csv)
    tags = model.predict(tf)
    
    return tags


# title
st.title("Stack Overflow Tags Assignment")

# Navigation

status = st.sidebar.radio("Navigation", ('Single', 'Collection'))
if (status == 'Collection'):
    
    csv_file = st.sidebar.file_uploader("CSV File",type = ".csv")    
    try:
        
        csv = pd.read_csv(csv_file)
        col_in = st.sidebar.selectbox("Please select column for post",list(csv.columns))
        
        if(st.sidebar.button('Submit')):
            post_list = csv[col_in]
            tags = get_collective(post_list)
            csv['PredictedTags'] = tags
            
            st.dataframe(csv)
            
            #data = csv.to_csv("Output.csv")
            st.download_button("Download Output", data = csv.to_csv().encode('utf-8'))
    
    except:
        st.error("Please make sure file is uploaded properly")
    
else:
    
    col1,col2=st.columns([1,1])
    col1.subheader("Post")
    col2.subheader("Tags")
    article = st.sidebar.text_area("Enter Post","")
    if(st.sidebar.button('Submit')):
        if article == "":
            st.error("Please Enter Post")
        else:
            category = get_category(article)
            col1.info(article)
            col2.text("Tag:{}".format(category[0]))
            