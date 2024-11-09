import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import pandas as pd

# Define image dimensions
img_height, img_width = 150, 150  # Adjust according to your model requirements
batch_size = 32

# Load Kannada to English mapping from an Excel file (optional)
excel_file = 'trans.xlsx'
df = pd.read_excel(excel_file)

# Create a mapping for Kannada characters and their meanings (optional)
kannada_to_english = dict(zip(df['Tulu_word'], df['English_Meaning']))  # Adjust column names as necessary

# Create ImageDataGenerator for loading training data
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    validation_split=0.2     # Split dataset into training and validation
)

# Create training and validation generators
train_generator = datagen.flow_from_directory(
    'D:\\Tulu_lipi\\dataset',  # Path to your training data
    target_size=(img_height, img_width),
    color_mode='grayscale',   # Use grayscale if your model expects it
    class_mode='categorical',  # Use categorical for multi-class classification
    batch_size=batch_size,
    subset='training',        # Set as training data
    shuffle=True,            # Shuffle data
)

validation_generator = datagen.flow_from_directory(
    'D:\\Tulu_lipi\\dataset',  # Path to your training data
    target_size=(img_height, img_width),
    color_mode='grayscale',
    class_mode='categorical',  # Use categorical for multi-class classification
    batch_size=batch_size,
    subset='validation',       # Set as validation data
    shuffle=True,
)

# Load the trained model if it exists
try:
    model = load_model('tulu_character_recognition_model2.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()  # Stop execution if model can't be loaded

# Load class indices from the training generator
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# Preprocess the drawn image for prediction
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((img_width, img_height))  # Resize
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.repeat(img_array, 3, axis=-1)  # Duplicate the channel to make it 3 channels
    img_array /= 255.0  # Normalize
    return img_array

# Streamlit app setup
st.title("Tulu Handwritten Character Recognition")
st.write("Draw a character in the box below, and the model will predict its Kannada equivalent.")

# Create a canvas component for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=10,
    stroke_color="#FFFFFF",  # White drawing color
    background_color="#000000",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

# If the user has drawn something, process and predict
if canvas_result.image_data is not None:
    # Convert canvas to an image
    drawn_image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype("uint8"), "RGB")
    
    # Preprocess the image for the model
    preprocessed_image = preprocess_image(drawn_image)
    
    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)
    
    # Retrieve the predicted character using the class mapping from generator
    predicted_character = index_to_class.get(predicted_class, "Unknown")
    
    # Display the result
    st.write(f"**Predicted Kannada Character**: {predicted_character}")

    # Optionally retrieve meaning from the dictionary if needed
    if predicted_character != "Unknown":
        meaning = kannada_to_english.get(predicted_character, "Meaning not found")
        st.write(f"**Meaning**: {meaning}")
