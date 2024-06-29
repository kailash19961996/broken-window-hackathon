import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Read the text file into a pandas DataFrame
df = pd.read_csv('comments.txt', delimiter=',', names=['timestamp', 'comment'])

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample by week to count comments
comments_per_week = df.resample('W').count()

# Resample by month to count comments
comments_per_month = df.resample('ME').count()

# Streamlit app
st.title('Comments Over Time')

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

with col1:
    st.subheader('Number of Comments per Week')
    fig_week, ax_week = plt.subplots(figsize=(5, 3))
    ax_week.plot(comments_per_week.index, comments_per_week['comment'], marker='o', label='Comments per Week')
    ax_week.set_xlabel('Time')
    ax_week.set_ylabel('Number of Comments')
    ax_week.set_title('Comments Per Week')
    ax_week.legend()
    ax_week.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_week)

with col2:
    st.subheader('Number of Comments per Month')
    fig_month, ax_month = plt.subplots(figsize=(5, 3))
    ax_month.plot(comments_per_month.index, comments_per_month['comment'], marker='o', label='Comments per Month')
    ax_month.set_xlabel('Time')
    ax_month.set_ylabel('Number of Comments')
    ax_month.set_title('Comments Per Month')
    ax_month.legend()
    ax_month.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_month)

# Display the raw data
st.subheader('Raw Data')
st.write(df)
