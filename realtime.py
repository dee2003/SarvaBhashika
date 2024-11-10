import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np
import pandas as pd
import pyttsx3
from gtts import gTTS
from io import BytesIO
import requests
import os
import zipfile

# Define image dimensions and paths
img_height, img_width = 150, 150
batch_size = 32
confidence_threshold = 0.6
model_path = 'tulu_character_recognition_model2.h5'
model_url = 'https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/tulu_character_recognition_model2.h5'
dataset_url = "https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/dataset.zip"
zip_file_path = "dataset.zip"
temp_dir = "temp_dataset"
dataset_path = os.path.join(temp_dir, "resize2")  # Adjust this if needed

# Load Excel file with translations
excel_file = 'words_translations.xlsx'
df = pd.read_excel(excel_file)

# Mappings for translations
kannada_to_english = dict(zip(df['Tulu_word'], df['English_Meaning']))
kannada_to_kannada = dict(zip(df['Tulu_word'], df['Kannada_Meaning']))
kannada_to_malayalam = dict(zip(df['Tulu_word'], df['Malayalam_Meaning']))
kannada_to_hindi = dict(zip(df['Tulu_word'], df['Hindi_Meaning']))

# Download and extract dataset if not already present
if not os.path.exists(zip_file_path):
    response = requests.get(dataset_url)
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    st.success("Dataset downloaded and extracted successfully!")

# Set up image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='training',
    shuffle=True,
    seed=42,
)

# Download model if not already present
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

# Load model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Class mappings
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# Function for image preprocessing
def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array /= 255.0
    return img_array

# Function to check if canvas is blank
def is_image_blank(image_data):
    return np.all(image_data[:, :, 0] == 0) or np.all(image_data[:, :, 0] == 255)

# Enhanced text-to-speech function
def speak(text, lang='en'):
    if lang == 'en':
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        tts = gTTS(text=text, lang=lang)
        audio_data = BytesIO()
        tts.write_to_fp(audio_data)
        st.audio(audio_data.getvalue(), format="audio/mp3")

# Floating tab with hover information
def floating_tab_with_hover():
    st.markdown("""
        <style>
            .floating-tab { position: fixed; bottom: 50px; right: 20px; background-color: #004085; color: white;
                padding: 10px; border-radius: 5px; cursor: pointer; z-index: 9999; width: 250px; font-family: Georgia, serif; }
            .hover-info { display: none; position: absolute; bottom: 45px; right: 0; background-color: #e0f7fa;
                color: #004085; padding: 10px; border-radius: 5px; width: 250px; z-index: 9998; font-family: Georgia, serif; }
            .floating-tab:hover .hover-info { display: block; }
        </style>
        <div class="floating-tab">
            💡 Did you know?
            <div class="hover-info">
                <p>Tulu features its unique script, Tigalari, which evolved from the Brahmi script. 
                Although Kannada dominates now, efforts to revive Tigalari are ongoing.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display the floating tab
floating_tab_with_hover()

# Instructions modal
def show_instructions():
    st.markdown("""
        <div style='background-color: #d1ecf1; padding: 20px; border-radius: 8px;'>
            <h2 style='color: #0c5460;'>How to Use the Drawing Tool</h2>
            <p>1. Select how many characters to draw.</p>
            <p>2. Drawing one character shows its Kannada equivalent.</p>
            <p>3. Drawing multiple characters shows their translations.</p>
        </div>
    """, unsafe_allow_html=True)

# Page header and instructions button
st.markdown("""
    <div style='background-color: #004085; padding: 15px; border-radius: 8px;'>
        <h1 style='text-align: center; color: #ffffff;'>VarnaMithra: Tulu Multilingual Translation</h1>
    </div>
""", unsafe_allow_html=True)
if st.button("🛈 Instructions"):
    show_instructions()

# Main drawing area for Tulu characters
character_count = st.selectbox("Select the number of characters to draw:", options=[1, 2, 3])
predictions = []

for i in range(character_count):
    st.write(f"Draw Character {i + 1}:")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=5,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=150,
        height=150,
        drawing_mode="freedraw",
        key=f"canvas_{i}",
    )

    if canvas_result.image_data is not None and not is_image_blank(canvas_result.image_data):
        drawn_image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"), "RGB")
        preprocessed_image = preprocess_image(drawn_image)
        predictions_array = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions_array)
        confidence = predictions_array[0][predicted_class]
        predictions.append(index_to_class.get(predicted_class, "Unknown") if confidence >= confidence_threshold else "Unrecognized")

# Display predictions and translations
if predictions:
    combined_characters = ''.join(predictions)
    st.markdown(f"<p>Predicted Characters: {combined_characters}</p>", unsafe_allow_html=True)
    
    if character_count > 1:
        translations = {
            "English": kannada_to_english.get(combined_characters, "Not found"),
            "Kannada": kannada_to_kannada.get(combined_characters, "Not found"),
            "Malayalam": kannada_to_malayalam.get(combined_characters, "Not found"),
            "Hindi": kannada_to_hindi.get(combined_characters, "Not found"),
        }
        for lang, meaning in translations.items():
            st.markdown(f"<p>{lang} Meaning: {meaning}</p>", unsafe_allow_html=True)
            if st.button(f"🔊 Read Aloud in {lang}"):
                speak(meaning, lang=lang[:2].lower())
