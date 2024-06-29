import streamlit as st
import requests
import pandas as pd
from geopy.geocoders import Nominatim
import pydeck as pdk

st.title('Maps')

# Function to get user's location based on IP address
def get_user_location():
    response = requests.get("https://ipinfo.io")
    data = response.json()
    location = data['loc'].split(',')
    latitude = float(location[0])
    longitude = float(location[1])
    return latitude, longitude

st.title('User Location on Map')

# Get the user's location
latitude, longitude = get_user_location()

# Display the user's location
st.write(f"Your location is: Latitude: {latitude}, Longitude: {longitude}")

# Create a DataFrame with the user's location
map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})

# Display the map
st.map(map_data)