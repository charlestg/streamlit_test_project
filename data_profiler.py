# https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

import streamlit as st
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import preprocessing
#from sklearn.datasets import load_boston
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

sns.set()
# requirement.txt (pip install )
# pandas_profiling streamlit_pandas_profiling

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(
    layout="wide",
)

# Display a title
st.sidebar.title('Data Profiler')

# upload files
data_file = st.sidebar.file_uploader("Upload CSV",type=["csv"])

# load the dataeset
#dataset = load_boston()
dataset = None
if data_file is not None:
    dataset = pd.read_csv(data_file)
  
if st.sidebar.checkbox('Pandas Profile') and dataset is not None:
    pr = ProfileReport(dataset, explorative=True, orange_mode=True)
    st_profile_report(pr)
    
if st.sidebar.checkbox('Head') and dataset is not None:
    st.table(dataset.head())
    
if st.sidebar.checkbox('Is Null') and dataset is not None:
    st.table( dataset.isnull().sum())
    

if st.sidebar.checkbox('Distribution Plot') and dataset is not None:
    labelIndex = 0
    label = st.sidebar.selectbox(
        'Select label:',
        dataset.columns,
        labelIndex
        )
        
    fig = plt.figure(figsize=(12,7))
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.distplot(dataset[label], bins=30)
    #plt.show()
    #st.pyplot(fig)
    st.pyplot(fig,use_container_width=True)
    
if st.sidebar.checkbox('Heat Map') and dataset is not None:
    st.sidebar.text("HeatMap 1: possitive correlations")
    colLen = len(dataset.columns)
    fig2 = plt.figure(figsize=(colLen, colLen))
    correlation_matrix = dataset.corr().round(2)
    # annot = True to print the values inside the square
    m = sns.heatmap(data=correlation_matrix, annot=True)
    st.pyplot(fig2)
    #st.plotly_chart(fig2,use_container_height=True, use_container_width=True)
    
if st.sidebar.checkbox('Interaction') and dataset is not None:
    st.sidebar.text("Feature vs Label")
    
