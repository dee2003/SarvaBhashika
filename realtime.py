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

# Define image dimensions
img_height, img_width = 150, 150
batch_size = 32
confidence_threshold = 0.6

# Load Excel file with additional columns for meanings
excel_file = 'words_translations.xlsx'
df = pd.read_excel(excel_file)

# Create mappings for meanings in English, Kannada, Malayalam, and Hindi
tulu_to_english = dict(zip(df['Tulu_word'], df['English_Meaning']))
tulu_to_kannada = dict(zip(df['Tulu_word'], df['Kannada_Meaning']))
tulu_to_malayalam = dict(zip(df['Tulu_word'], df['Malayalam_Meaning']))
tulu_to_hindi = dict(zip(df['Tulu_word'], df['Hindi_Meaning']))

model_url = 'https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/tulu_character_recognition_model2.h5'
dataset_url = 'https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/dataset.zip'
model_path = 'tulu_character_recognition_model2.h5'
dataset_zip_path = 'dataset.zip'
dataset_dir = 'dataset'  # Directory to extract dataset contents

# Function to download a file
def download_file(url, path):
    response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Check if model exists, otherwise download
if not os.path.exists(model_path):
    st.info("Downloading model, please wait...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.success("Model downloaded successfully!")

# Load model with error handling
try:
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error("An error occurred while loading the model.")
    st.text(f"Error details: {e}")
# Download and extract dataset if not already extracted
if not os.path.exists(dataset_dir):
    if not os.path.exists(dataset_zip_path):
        st.info("Downloading dataset, please wait...")
        download_file(dataset_url, dataset_zip_path)
        st.success("Dataset downloaded successfully!")

    if zipfile.is_zipfile(dataset_zip_path):
        st.info("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        st.success("Dataset extracted successfully!")
    else:
        st.error("The dataset file is corrupted or not a valid zip file.")

# Verify dataset structure
if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
    # Ensure there are subdirectories in dataset_dir for each class
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if subdirs:
        st.success("Dataset structure verified.")

        # Set up ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1./255)
        try:
            train_generator = datagen.flow_from_directory(
                dataset_dir,
                target_size=(150, 150),  # Adjust target size as per model input
                batch_size=32,
                class_mode='categorical'
            )
            st.success("Data generator created successfully.")
        except Exception as e:
            st.error(f"Error creating data generator: {e}")
    else:
        st.error("The dataset directory does not contain class subdirectories.")
else:
    st.error("Dataset directory is empty or does not exist.")
    
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array /= 255.0
    return img_array

def is_image_blank(image_data):
    return np.all(image_data[:, :, 0] == 0) or np.all(image_data[:, :, 0] == 255)

# Enhanced speak function with gTTS for non-English languages
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

# Function to add a floating tab with hover info
def floating_tab_with_hover():
    # Custom CSS for floating tab and hover effect
    st.markdown(
    """
    <style>
        .floating-tab {
            position: fixed;
            bottom: 50px;  /* Adjusted position to ensure visibility below heading */
            right: 20px;
            background-color: #004085;  /* Dark blue background */
            color: white;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 250px;
            font-family: Georgia, serif; /* Width to ensure enough space for the text */
        }
        .hover-info {
            display: none;
            position: absolute;
            bottom: 45px; /* Positioning to show above the button */
            right: 0;
            background-color: #e0f7fa;  /* Light blue background for hover info */
            color: #004085;  /* Dark blue text for visibility */
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            width: 250px;
            z-index: 9998;
            font-family: Georgia, serif;/* Ensure hover info is above other elements */
        }
        .floating-tab:hover .hover-info {
            display: block;
        }
    </style>
    """,
    unsafe_allow_html=True
)

    
    # Floating tab with hover info
st.markdown(
    """
    <div class="floating-tab">
        💡 Did you know?
        <div class="hover-info">
            <p>Tulu features its unique script, Tigalari, which evolved from the ancient Brahmi script and differs from Kannada and Malayalam. 
            Although Kannada has become dominant in writing, there are ongoing efforts to revive Tigalari and promote its cultural significance among the youth.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Call the function to display the floating tab
floating_tab_with_hover()

# Instructions modal
def show_instructions():
    st.markdown(""" 
    <div style='background-color: #d1ecf1; padding: 20px; border-radius: 8px; font-family: Georgia;'> 
        <h2 style='color: #0c5460; font-size: 1.3em;'>How to Use the Drawing Tool</h2> 
        <p style='color: #0c5460; font-size: 1.1em;'>1. Select how many characters of a word to draw.</p> 
        <p style='color: #0c5460; font-size: 1.1em;'>2. Drawing one character shows its Kannada equivalent.</p> 
        <p style='color: #0c5460; font-size: 1.1em;'>3. Drawing two or three characters displays their Kannada equivalents and the translation of the word in multiple languages.</p> 
        <p style='color: #0c5460; font-size: 1.1em;'><strong style='color: #004085;'>Enjoy translating your Tulu words!</p> 
    </div> """, unsafe_allow_html=True)


st.markdown(
    """
    <div style='background-color: #004085; padding: 15px; border-radius: 8px; margin-bottom: 20px;'> 
        <h1 style='text-align: center; color: #ffffff; font-size: 2.5em;'>VarnaMithra: Multilingual Translation for Tulu</h1> 
        <p style='text-align: center; color: #e0e0e0; font-size: 1.2em;font-family: "Georgia", serif; font-style: italic;'> "Bringing Tulu to Life: Translate, Speak, and Discover a World of Languages!"</p>
    </div> """, unsafe_allow_html=True)


# Show buttons for instructions and fun fact
if st.button("🛈 Instructions"):
    show_instructions()


# Select number of characters
character_count = st.selectbox("Select the number of characters to draw:", options=[1, 2, 3], index=0)
predictions = []

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas"
)

# Process the drawn image
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    
    # Check for blank image (i.e., no drawing)
    if not is_image_blank(canvas_result.image_data):
        st.image(img, caption="Your Drawing", use_column_width=True)
        img_array = preprocess_image(img)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = index_to_class[predicted_class_idx]
        
        # Prediction output
        st.markdown(f"Prediction: **{predicted_class}**")
        
        # Display translations
        if predicted_class in tulu_to_english:
            st.write(f"English: {tulu_to_english[predicted_class]}")
        if predicted_class in tulu_to_kannada:
            st.write(f"Kannada: {tulu_to_kannada[predicted_class]}")
        if predicted_class in tulu_to_malayalam:
            st.write(f"Malayalam: {tulu_to_malayalam[predicted_class]}")
        if predicted_class in tulu_to_hindi:
            st.write(f"Hindi: {tulu_to_hindi[predicted_class]}")

        # Speaking the translated word in multiple languages
        speak(tulu_to_english.get(predicted_class, ''), lang='en')
        speak(tulu_to_kannada.get(predicted_class, ''), lang='kn')
        speak(tulu_to_malayalam.get(predicted_class, ''), lang='ml')
        speak(tulu_to_hindi.get(predicted_class, ''), lang='hi')
else:
    st.write("Draw something on the canvas!")
