import streamlit as st
import tempfile
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
import requests
import zipfile

# Define image dimensions and paths
img_height, img_width = 150, 150
batch_size = 32
confidence_threshold = 0.6
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

# Define model URL and local model path
model_url = "https://github.com/dee2003/Varnamitra-Tulu-word-translation/releases/download/v1.0/tulu_character_recognition_model2.h5"
model_path = "tulu_character_recognition_model2.h5"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    try:
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading the model: {e}")
        st.stop()

# Load the model
try:
    model = load_model(model_path)
    print("success")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}


# Define a function to preprocess the image
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

def speak(text, lang='en'):
    if lang == 'en':
        # Use gTTS for both English and other languages to ensure compatibility
        tts = gTTS(text=text, lang='en')
    else:
        # For other languages
        tts = gTTS(text=text, lang=lang)
    
    # Save audio to a BytesIO buffer
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    
    # Reset the pointer to the beginning of the BytesIO buffer
    audio_data.seek(0)
    
    # Play audio in Streamlit
    st.audio(audio_data, format="audio/mp3")


# Function to add a floating tab with hover info
col1, col2 = st.columns([3,2])

with col1:
    
    st.markdown(
        """
        <div style='background-color: #004085; padding: 5px 15px; border-radius: 8px; text-align: left; margin-bottom: 20px;'>
            <h1 style='color: #ffffff; font-size: 3em;'>SarvaBhashika: Translating Tulu to diverse languages</h1>
            <p style='color: #e0e0e0; font-size: 1.3em;font-family: "Georgia", serif; font-style: italic;'>"Bringing Tulu to Life: Translate, Speak, and Discover a World of Languages!"</p>
        </div>
        """, unsafe_allow_html=True
    )

  

# Instructions section
    st.markdown("""
<div style='background-color: #d1ecf1; padding: 8px; border-radius: 8px; font-family: Georgia; font-style: italic; margin-bottom: 10px;'>
    <p style='color: #0c5460; font-size: 1.1em; margin: 2px 0;'>Draw one character to see its Kannada equivalent, or draw multiple characters of a word to get translations in Kannada and other languages.</p>
</div>
""", unsafe_allow_html=True)



    character_count = st.selectbox("Select the number of characters to draw:", options=[1, 2, 3], index=0)
    predictions = []
    columns = st.columns(character_count)

    for i in range(character_count):
        with columns[i]:
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

# Displaying the final combined translations with speaker icon for each language.
if predictions:
    combined_characters = ''.join(predictions)
    st.markdown(f"<p style='font-size:25px; color:Blue; font-weight:bold;'>Predicted Kannada Characters: {combined_characters}</p>", unsafe_allow_html=True)
    
    # Fetching translations
    english_meaning = kannada_to_english.get(combined_characters, "Meaning not found")
    kannada_meaning = kannada_to_kannada.get(combined_characters, "Meaning not found")
    malayalam_meaning = kannada_to_malayalam.get(combined_characters, "Meaning not found")
    hindi_meaning = kannada_to_hindi.get(combined_characters, "Meaning not found")

    # Translation layout with 2 columns each for left and right side
    left_col, right_col = st.columns([1, 1])

    # First two translations on the left
    with left_col:
        # English Translation
        st.markdown(
            f"""
            <div style='padding: 10px; border-radius: 5px; margin: 10px 0; background-color: #d1e7dd;'>
                <p style='font-size:20px; color:#0f5132; font-weight:bold;'>English Meaning: {english_meaning}</p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔊 Read Aloud in English", key="en_read_aloud"):
            speak(english_meaning, lang="en")
        
        # Kannada Translation
        st.markdown(
            f"""
            <div style='padding: 10px; border-radius: 5px; margin: 10px 0; background-color: #fff3cd;'>
                <p style='font-size:20px; color:#856404; font-weight:bold;'>Kannada Meaning: {kannada_meaning}</p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔊 Read Aloud in Kannada", key="kn_read_aloud"):
            speak(kannada_meaning, lang="kn")

    # Next two translations on the right
    with right_col:
        # Malayalam Translation
        st.markdown(
            f"""
            <div style='padding: 10px; border-radius: 5px; margin: 10px 0; background-color: #f8d7da;'>
                <p style='font-size:20px; color:#721c24; font-weight:bold;'>Malayalam Meaning: {malayalam_meaning}</p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔊 Read Aloud in Malayalam", key="ml_read_aloud"):
            speak(malayalam_meaning, lang="ml")
        
        # Hindi Translation
        st.markdown(
            f"""
            <div style='padding: 10px; border-radius: 5px; margin: 10px 0; background-color: #cce5ff;'>
                <p style='font-size:20px; color:#004085; font-weight:bold;'>Hindi Meaning: {hindi_meaning}</p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("🔊 Read Aloud in Hindi", key="hi_read_aloud"):
            speak(hindi_meaning, lang="hi")




with col2:
    url = "https://raw.githubusercontent.com/dee2003/SarvaBhashika/main/chart.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))


    target_height = 500 # Increase for larger display
    aspect_ratio = img.width / img.height
    target_width = int(target_height * aspect_ratio)
    resized_img = img.resize((target_width, target_height))

    st.image(resized_img, caption="Tulu-Kannada Character Mapping Chart", use_container_width=True)
