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
import streamlit as st



# Define image dimensions
img_height, img_width = 150, 150
batch_size = 32
confidence_threshold = 0.6

# Load Excel file with additional columns for meanings
excel_file = 'words_translations.xlsx'
df = pd.read_excel(excel_file)

# Create mappings for meanings in English, Kannada, Malayalam, and Hindi
kannada_to_english = dict(zip(df['Tulu_word'], df['English_Meaning']))
kannada_to_kannada = dict(zip(df['Tulu_word'], df['Kannada_Meaning']))
kannada_to_malayalam = dict(zip(df['Tulu_word'], df['Malayalam_Meaning']))
kannada_to_hindi = dict(zip(df['Tulu_word'], df['Hindi_Meaning']))
# URL of the dataset from the GitHub release
dataset_url = "https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/dataset.zip"

# File path to save the downloaded dataset
zip_file_path = "dataset.zip"

# Download the dataset
if not os.path.exists(zip_file_path):
    response = requests.get(dataset_url)
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    st.success("Dataset downloaded successfully!")

# Unzip the dataset
temp_dir = "temp_dataset"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Set the path to the unzipped dataset
dataset_path = os.path.join(temp_dir, "resize2")  # Adjust this to the correct folder name

# Load model and generator setup
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


model_path = 'tulu_character_recognition_model2.h5'
model_url = 'https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/tulu_character_recognition_model2.h5'
model = load_model(model_url)


try:
    # Check if the model file exists before trying to load
    if not os.path.exists(model_url):
        raise FileNotFoundError(f"Model file does not exist at path: {model_path}")
    
    model = load_model(model_path)
    st.success("Model loaded successfully.")
    print("Model loaded successfully.")  # Logging to console for debugging
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    print(f"Error: {e}")  # Log the error to console
except Exception as e:
    st.error(f"Error loading the model: {e}")
    print(f"Error loading the model: {e}")  # Log the error to conso
    
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
    </div>
    """, unsafe_allow_html=True)




st.markdown(
    """
    <div style='background-color: #004085; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <h1 style='text-align: center; color: #ffffff; font-size: 2.5em;'>VarnaMithra: Multilingual Translation for Tulu</h1>
        <p style='text-align: center; color: #e0e0e0; font-size: 1.2em;font-family: "Georgia", serif; font-style: italic;'>"Bringing Tulu to Life: Translate, Speak, and Discover a World of Languages!"</p>
    </div>
    """, unsafe_allow_html=True
)


# Show buttons for instructions and fun fact
if st.button("🛈 Instructions"):
    show_instructions()



# Select number of characters
character_count = st.selectbox("Select the number of characters to draw:", options=[1, 2, 3], index=0)
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

    if canvas_result.image_data is not None:
        if not is_image_blank(canvas_result.image_data):
            drawn_image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"), "RGB")
            preprocessed_image = preprocess_image(drawn_image)
            predictions_array = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions_array)
            confidence = predictions_array[0][predicted_class]
            if confidence >= confidence_threshold:
                predicted_character = index_to_class.get(predicted_class, "Unknown")
                predictions.append(predicted_character)
            else:
                predictions.append("Unrecognized")

if predictions:
    combined_characters = ''.join(predictions)

    # Display predicted character(s) if only one character is selected, without translations
    if character_count == 1:
        st.markdown(f"<p style='font-size:25px; color:Blue; font-weight:bold;'>Predicted Character: {combined_characters}</p>", unsafe_allow_html=True)
    
    # If more than one character, display translations as well
    else:
        english_meaning = kannada_to_english.get(combined_characters, "Meaning not found")
        kannada_meaning = kannada_to_kannada.get(combined_characters, "Meaning not found")
        malayalam_meaning = kannada_to_malayalam.get(combined_characters, "Meaning not found")
        hindi_meaning = kannada_to_hindi.get(combined_characters, "Meaning not found")

        # Display predicted characters
        st.markdown(f"<p style='font-size:25px; color:Blue; font-weight:bold;'>Predicted Kannada Characters: {combined_characters}</p>", unsafe_allow_html=True)

        st.markdown(
        f"""
        <div style='background-color: #d1e7dd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
            <p style='font-size:20px; color:#0f5132; font-weight:bold;'>English Meaning: {english_meaning}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔊 Read Aloud in English", key="en_read_aloud"):
            speak(english_meaning, lang='en')

        st.markdown(
            f"""
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <p style='font-size:20px; color:#856404; font-weight:bold;'>Kannada Meaning: {kannada_meaning}</p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🔊 Read Aloud in Kannada", key="kn_read_aloud"):
            speak(kannada_meaning, lang='kn')

        st.markdown(
            f"""
            <div style='background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <p style='font-size:20px; color:#721c24; font-weight:bold;'>Malayalam Meaning: {malayalam_meaning}</p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🔊 Read Aloud in Malayalam", key="ml_read_aloud"):
            speak(malayalam_meaning, lang='ml')

        st.markdown(
            f"""
            <div style='background-color: #cce5ff; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <p style='font-size:20px; color:#004085; font-weight:bold;'>Hindi Meaning: {hindi_meaning}</p>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🔊 Read Aloud in Hindi", key="hi_read_aloud"):
            speak(hindi_meaning, lang='hi')
