import streamlit as st
import os
from datetime import datetime
from PIL import Image

# Create a directory to store files if it doesn't exist
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded_files', uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

st.title('Image and Comment Uploader')

# File uploader allows user to add file
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Save the file
    if save_uploaded_file(uploaded_file):
        st.success(f'Saved file: {uploaded_file.name} in uploaded_files')
    else:
        st.error('Failed to save file')

# Text box for comments
comment = st.text_area("Comment here...")
if st.button('Submit Comment'):
    # Save the comment with a timestamp
    if comment:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(os.path.join('uploaded_files', f'comment_{timestamp}.txt'), "w") as f:
            f.write(comment)
        st.success('Comment saved successfully.')
    else:
        st.error('Please enter a comment.')

st.markdown("---")
st.info('All images and comments are saved in the local "uploaded_files" directory.')
