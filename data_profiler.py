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

dataset = None
#dataset_numeric = None
columns_numeric = []
# Enter file
data_path = st.sidebar.text_input("Url", 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

if len(data_path) > 0:
    dataset = pd.read_csv(data_path)

# upload files
data_file = st.sidebar.file_uploader("Upload CSV",type=["csv"])

# load the dataeset
#dataset = load_boston()

if data_file is not None:
    dataset = pd.read_csv(data_file)
    
if dataset is not None:
    #dataset_numeric = dataset.select_dtypes('number')
    columns_numeric = dataset.select_dtypes('number').columns

label = st.sidebar.selectbox(
        'Select label:',
        columns_numeric,
        0
        )
if st.sidebar.checkbox('Pandas Profile') and dataset is not None:
    pr = ProfileReport(dataset, explorative=True, orange_mode=True)
    st_profile_report(pr)
 
if st.sidebar.checkbox('Describe') and dataset is not None:
    st.table(dataset.describe())
    
if st.sidebar.checkbox('Head') and dataset is not None:
    st.table(dataset.head())
    
if st.sidebar.checkbox('Null Counts') and dataset is not None:
    st.table( dataset.isnull().sum())
    
# dist plot 
if st.sidebar.checkbox('Distribution Plot') and dataset is not None:
    labelIndex = 0
    columnName = st.sidebar.selectbox(
        'Select Column:',
        columns_numeric,
        labelIndex
        )
        
    fig = plt.figure(figsize=(12,7))
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.distplot(dataset[columnName], bins=30)
    #plt.show()
    #st.pyplot(fig)
    st.pyplot(fig,use_container_width=True)
    
if st.sidebar.checkbox('Distribution Plot All') and dataset is not None:
    #plt.figure(figsize=(20, 5))
    n_cols = 5
    n_rows = len(columns_numeric) // n_cols + 1  if len(columns_numeric) % n_cols != 0 else len(columns_numeric) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols) # figsize=(n_rows*2,n_cols)
    axes = axes.flatten()
    #fig, axes = plt.subplots(ncols=len(columns_numeric), figsize=(30,15))
    for i, col in zip(axes, columns_numeric):
      sns.distplot(dataset[col], ax=i)
      plt.tight_layout() 
      #sns.distplot(dataset[col],ax=axes[i//n_cols,i%n_cols])
    #st.pyplot(fig,use_container_width=True)
    st.pyplot(fig) 
    
if st.sidebar.checkbox('Corelation Plot All') and dataset is not None:
    st.header(label)
    n_cols = 5
    n_rows = len(columns_numeric) // n_cols + 1  if len(columns_numeric) % n_cols != 0 else len(columns_numeric) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols) 
    axes = axes.flatten()
    for i, col in zip(axes, columns_numeric):
        if col == label:
            continue
        sns.scatterplot(data=dataset, x=col,y=label, ax=i)
        plt.tight_layout() 
    st.pyplot(fig)

   
# heatmap 
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