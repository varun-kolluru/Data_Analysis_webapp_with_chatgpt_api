import streamlit as st
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from pandasai import PandasAI
import os
from pandasai.llm.openai import OpenAI

st.title("Data Analyser And Visualiser")
st.write("Tool which made data analysis easy. Now You can upload your excel or csv file and can make your analysis")
file_type=st.selectbox(label="Select your input File Type:",options=["CSV","Excel"])
file=st.file_uploader("Upload")

df=[]

st.sidebar.subheader("You can choose filters here")
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.sidebar.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            to_filter_values1= st.sidebar.multiselect("Type values to be present in "+str(column),df[column].unique())
            if to_filter_values1:
                df=df[df[column].map(lambda x:x in to_filter_values1)]
            to_filter_values2= st.sidebar.multiselect("Type values to be not present "+str(column),df[column].unique())
            if to_filter_values2:
                df=df[df[column].map(lambda x:x not in to_filter_values2)]
            if is_numeric_dtype(df[column]):
                _min,_max=min(df[column]),max(df[column])
                start,end=st.sidebar.slider("select a between range",_min,_max,(_min,_max),step=1,)
                df=df[df[column].map(lambda x:start<=x<=end)]
    return df

def chat_gpt(text,key):
    os.environ['OPENAI_API_KEY']=key
    llm=OpenAI(temperature=0)
    pandas_ai=PandasAI(llm=llm,conversational=True)
    return pandas_ai.run(df,prompt=text)


if file:
    if file_type=="CSV":
        df=pd.read_csv(file)
    elif file_type=="Excel":
        df=pd.read_excel(file)
else:
    df=pd.read_csv("C:/Users/varun/Documents/vscode python/USA_cars_datasets.csv")

st.header("Your Data")

data_loaded=False
if len(df)!=0:
    df1=[]
    df1=filter_dataframe(df)
    st.write(df1)
    st.write("records dropped during filtering:-",len(df)-len(df1))
    st.header("Stats")
    st.write("Total rows Present",len(df1))
    st.write("Current Data set General description",df1.describe())
    data_loaded=True
    st.download_button("Download as csv",df1.to_csv(index=False).encode('utf-8'),'file.csv','text/csv')


def plot(g_type,x=None,y=None,z=None):
    if g_type=='bar':
        fig = px.bar(df1,x=x,y=y)
    elif g_type=='pie':
        fig= px.pie(data_frame=y,names =x)
    elif g_type=='scatter':
        fig = px.scatter(df1,x=x,y=y)
    elif g_type=='line':
        fig = px.line(df1,x=x,y=y)
    elif g_type=='scatter3d':
        fig = px.scatter_3d(df1,x=x,y=y,z=z)
    elif g_type=='line3d':
        fig=px.line_3d(df1,x=x,y=y,z=z)
    return st.plotly_chart(fig,use_container_width=True, height = 200)

st.header("chat with your Data?")
key=st.text_input("enter your openAi Api key",type='password')
text=st.text_input(label='Ask Bot anything about your data')
if key:
    message= st.chat_message("user")
    message.write(text)
    if text:
        message = st.chat_message("ai")
        message.write(chat_gpt(text,key))


    
st.header("Data Visualization")
g=x=y=z=None
g=st.selectbox(label="choose graph type",options=['bar','pie','scatter','line','scatter3d','line3d'])
if g=='bar':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    y=st.selectbox(label='Y value',options=[None]+[i for i in df1.columns])
    if x!=None and y!=None:
        plot(g,x,y)
elif g=='pie':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    if x!=None:
        tmp=pd.DataFrame(columns=[x])
        tmp[x]=df1[x]
        print(tmp)
        if len(tmp)!=0:
            plot(g,x,tmp)
elif g=='scatter':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    y=st.selectbox(label='Y value',options=[None]+[i for i in df1.columns])
    if x!=None and y!=None:
        plot(g,x,y)
elif g=='line':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    y=st.selectbox(label='Y value',options=[None]+[i for i in df1.columns])
    if x!=None and y!=None:
        plot(g,x,y)
elif g=='scatter3d':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    y=st.selectbox(label='Y value',options=[None]+[i for i in df1.columns])
    z=st.selectbox(label='Z value',options=[None]+[i for i in df1.columns])
    if x!=None and y!=None and z!=None:
        plot(g,x,y,z)
elif g=='line3d':
    x=st.selectbox(label='X value',options=[None]+[i for i in df1.columns])
    y=st.selectbox(label='Y value',options=[None]+[i for i in df1.columns])
    z=st.selectbox(label='Z value',options=[None]+[i for i in df1.columns])
    if x!=None and y!=None and z!=None:
        plot(g,x,y,z)






    







