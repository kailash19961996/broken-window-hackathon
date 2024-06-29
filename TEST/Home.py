# main.py

import streamlit as st
import os
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import openai
import requests
import pandas as pd
from geopy.geocoders import Nominatim

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

# Set OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = api_key

# Define categories
categories = [
    "graffiti",
    "garbage",
    "broken_window",
    "green_spaces",
    "public_buildings",
    "sports_and_social_events"
]

# Directory setup for each category
base_dir = 'classified_images'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
for category in categories:
    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    comments_path = os.path.join(category_dir, 'comments.txt')
    if not os.path.exists(comments_path):
        with open(comments_path, 'w') as f:
            pass

# Function to classify image
def classify_image(image):
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    category_index = probs.argmax().item()
    return categories[category_index]

# Function to get user's location based on IP address
def get_user_location():
    response = requests.get("https://ipinfo.io")
    data = response.json()
    location = data['loc'].split(',')
    latitude = float(location[0])
    longitude = float(location[1])
    return latitude, longitude

# Function to save location data
def save_location(category_dir, latitude, longitude):
    location_path = os.path.join(category_dir, 'locations.csv')
    if not os.path.exists(location_path):
        with open(location_path, 'w') as f:
            f.write("latitude,longitude\n")
    with open(location_path, 'a') as f:
        f.write(f"{latitude},{longitude}\n")

# Function to summarize comments
def summarize_comments():
    summaries = []
    feedback_counts = []
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        comments_path = os.path.join(category_dir, 'comments.txt')
        
        # Count number of feedbacks (images) in the category folder
        feedback_count = len([name for name in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, name)) and name != 'comments.txt'])
        feedback_counts.append((category, feedback_count))
        
        with open(comments_path, 'r') as file:
            comments = file.read()
        if comments:
            prompt = f"Summarize the following user comments and address the major concerns mentioned:\n\n{comments}"
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.5
            )
            summary = response.choices[0].message['content'].strip()
            summaries.append((category, summary))
    
    # Sort summaries based on feedback count from highest to lowest
    feedback_counts.sort(key=lambda x: x[1], reverse=True)
    
    summary_text = ""
    for category, count in feedback_counts:
        summary_text += f"**{category}** - {count} feedback(s)\n"
        summary_text += next((summary for cat, summary in summaries if cat == category), "No summary available") + "\n\n"
    
    return summary_text



# # Streamlit page
# st.set_page_config(
#         page_title="Neighbourhood Report",
#         page_icon="👋",
#     )

# Streamlit UI
st.title('Image and Comment Uploader')
# st.title("Neighbourhood Report 👋")

st.sidebar.page_link("pages/Report.py", label="Reports", icon="👋")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Classify and save the image
    category = classify_image(image)
    category_dir = os.path.join(base_dir, category)
    image_path = os.path.join(category_dir, uploaded_file.name)
    image.save(image_path)
    st.success(f'Classified as {category} and saved to {category_dir}')
    
    # Get and save the user's location
    latitude, longitude = get_user_location()
    save_location(category_dir, latitude, longitude)
    st.write(f"Location saved: Latitude: {latitude}, Longitude: {longitude}")
    
    # Text box for comments
    comment = st.text_input("Add a comment about the image:")
    if st.button('Submit Comment'):
        if comment:
            with open(os.path.join(category_dir, 'comments.txt'), 'a') as f:
                f.write(f"{datetime.now()}: {comment}\n")
            st.success('Comment added successfully!')
        else:
            st.error('Please add a comment before submitting.')

# Summarize comments
if st.button('Summarize Comments'):
    summaries = summarize_comments()
    st.markdown(summaries)
