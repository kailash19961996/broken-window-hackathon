import streamlit as st


st.set_page_config(
        page_title="Weekly Report",
        page_icon="👋",
    )

st.title("Neighbourhood Report 👋")
st.sidebar.page_link("pages/weekly_report.py", label="Weekly Report", icon="👋")