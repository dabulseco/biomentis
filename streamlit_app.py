import streamlit as st

from biomni.agent import A1


@st.cache_resource
def get_agent():
    return A1(path="./data", use_tool_retriever=False, expected_data_lake_files=[])


get_agent().launch_streamlit_demo()
