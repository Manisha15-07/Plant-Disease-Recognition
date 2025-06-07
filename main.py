import streamlit as st
import numpy as np
import base64
import requests
import pandas as pd
import joblib
from PIL import Image
from streamlit_option_menu import option_menu
import process

# Set page configuration
st.set_page_config("Plants", page_icon=":seedling:", layout="wide")

# Background image
with open('agro12.jpg', 'rb') as f:
    data = f.read()
imgs = base64.b64encode(data).decode()
css = f"""
    <style>
    [data-testid="stAppViewContainer"]{{
        background-image: url('data:image/png;base64,{imgs}');
        background-size:cover;
    }}
    [data-testid="stSidebar"]{{
        background-color:#becc58;
    }}
    </style>
"""
st.markdown(css, unsafe_allow_html=True)

# Main app logic
def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Plant Disease Recognition System",
            menu_icon="leaf",
            options=["Home", "Plant Diseases", "Plant Identification", "Weather", "Crop Yield", "About"],
            icons=["house-door", "search", "tree", "sun", "bar-chart-line", "info-circle"],
            default_index=0
        )

    if selected == "Home":
        st.title("🌿 Welcome to the Plant Disease Recognition System")
        st.write("Explore the features using the sidebar menu.")
        st.markdown("""
        👨‍🌾 This system uses **AI and Machine Learning** to help farmers, researchers, and agriculturists:

    - 🦠 **Detect plant diseases** from leaf images.
    - 🌳 **Identify plant species** using camera or uploaded photos.
    - ☀️ **Check weather forecasts** to plan ahead.
    - 🌾 **Predict crop yields** using environmental and input data.

    🧭 Use the **sidebar menu** on the left to explore each feature.

    🛠️ Built with ❤️ using **Streamlit + Python + Machine Learning**.
    """)

    elif selected == "Plant Diseases":
        st.header("Plant Disease Detection")
        detection_choice = st.radio("Detection Method", ["Upload Image", "Camera Input"])
        uploaded_image = None

        if detection_choice == "Upload Image":
            uploaded_image = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
        elif detection_choice == "Camera Input":
            uploaded_image = st.camera_input("📷 Capture a leaf image")

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Input Image", use_column_width=True)
            st.write("Processing image for disease detection...")

            if st.button("🚀 Submit"):
                try:
                    x = process.imageAns(uploaded_image)
                    classes = ['Healthy', 'Powdery', 'Rust']
                    ans = np.argmax(x)
                    st.subheader(f"Prediction: {classes[ans]}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    elif selected == "Plant Identification":
        st.header("🌳 Identify Plant Species")
        uploaded_image = st.file_uploader("📤 Upload a plant image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Processing image for plant identification...")

            if st.button("🚀 Submit"):
                try:
                    x = process.imageAnsId(uploaded_image)
                    classes = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn',
                               'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale',
                               'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peperchili',
                               'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweetpotatoes',
                               'tobacco', 'waterapple', 'watermelon']
                    ans = np.argmax(x)
                    st.subheader(f"Identified Plant: {classes[ans]}")
                except Exception as e:
                    st.error(f"Error during identification: {e}")

    elif selected == "Weather":
        st.title("☀️ Weather Forecast")
        city_name = st.text_input("🏙️ Enter City Name")

        if city_name:
            try:
                API_KEY = "9b99c2520d9ddb36ed867de4196e0ede"
                api_address = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={API_KEY}"
                res = requests.get(api_address)
                data = res.json()

                for i in data['list'][:5]:  # Show only the first 5 entries
                    st.subheader(i['dt_txt'])
                    date_obj = pd.to_datetime(i['dt_txt']).day_name()
                    st.write(f"Day: {date_obj}")
                    col1, col2 = st.columns(2)
                    with col1:
                        img = 'rain.jpg' if i['weather'][0]['description'] == "light rain" else 'cloud.jpg'
                        st.image(img, width=100)
                    with col2:
                        st.write(f"Description: {i['weather'][0]['description']}")
                        st.write(f"Temperature: {round(i['main']['temp'] - 273.15, 2)} °C")
                        st.write(f"Wind Speed: {i['wind']['speed']} km/hr")
                    st.divider()
            except Exception as e:
                st.error(f"Error fetching weather: {e}")

    elif selected == "Crop Yield":
        st.title("Crop Yield Prediction")
        col1, col2, col3 = st.columns(3)

        with col1:
            crop = st.text_input("Crop")
            crop_year = st.number_input("Crop Year", step=1)
            season = st.text_input("Season")

        with col2:
            state = st.text_input("State")
            area = st.number_input("Area (in hectares)")
            production = st.number_input("Production (in tonnes)")

        with col3:
            annual_rainfall = st.number_input("Annual Rainfall (mm)")
            fertilizer = st.number_input("Fertilizer Used (kg)")
            pesticide = st.number_input("Pesticide Used (kg)")

        if st.button("🚀 Submit"):
            try:
                crop_encoder = joblib.load('crop_enc.pkl')
                season_encoder = joblib.load('season_enc.pkl')
                state_encoder = joblib.load('state_enc.pkl')

                features = [
                    crop_encoder.transform([crop])[0],
                    crop_year,
                    season_encoder.transform([season])[0],
                    state_encoder.transform([state])[0],
                    area,
                    production,
                    annual_rainfall,
                    fertilizer,
                    pesticide
                ]

                model = joblib.load('crop.pkl')
                prediction = model.predict([features])
                st.subheader(f"Predicted Crop Yield: {prediction[0]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    elif selected == "About":
        st.title("About This Project")
        st.markdown("""
    ### 🌾 Plant Disease Recognition System

    This project is a smart agriculture assistant designed to help **farmers, researchers, and agronomists** manage crops more effectively using **Artificial Intelligence**.

    ---
    ### 🔍 Key Features:
    - 🦠 **Plant Disease Detection**  
      Upload or capture leaf images to detect diseases like *Powdery Mildew* or *Rust*.
    
    - 🌳 **Plant Identification**  
      Instantly identify plant species from photos using image classification.
    
    - ☀️ **Weather Forecasting**  
      Get 5-day weather forecasts to plan irrigation and harvesting.
    
    - 📈 **Crop Yield Prediction**  
      Estimate potential yield based on factors like rainfall, area, and input usage.

    ---
    ### 🛠️ Technologies Used:
    - 🐍 Python
    - 🤖 Machine Learning (Classification Models)
    - 📦 Streamlit for the user interface
    - 🌐 OpenWeatherMap API for weather data

    ---
    ### 🌍 Goal:
    To support **sustainable agriculture** by providing real-time insights that help:
    - Increase productivity 🌿
    - Reduce crop losses 💧
    - Improve decision-making 🧠
    - Ensure food security 🌾

    ---

    🧑‍💻 **Developed with passion to empower modern farming!**
    """)

if __name__ == "__main__":
    main()
