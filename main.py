import pandas as pd
import numpy as np
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEYS = os.getenv("OPENAI_KEYS")

from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

os.environ['OPENAI_API_KEY'] = OPENAI_KEYS

st.title('Chat with the Database 💬')
file = st.file_uploader("Upload a CSV file", type=["csv"])

data = pd.DataFrame(columns=['Date', 'Time', 'Query', 'Answer'])

def query(unique_key):
    query = st.text_input("Enter your query here", key=f"query_{unique_key}")
    if query:
        st.write("Your query is: ", query)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)
        answer = agent.run(query)  
        st.write("Answer:", answer)

if file is not None:
    with st.expander("Show Database"):  
        df = pd.read_csv(file, encoding = "ISO-8859-1")
        st.dataframe(df.head())
        data = df.copy()
    
    with st.expander("Describe Database"):
        st.dataframe(data.describe())
    
    query(0)
