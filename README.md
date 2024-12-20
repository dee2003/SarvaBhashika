#SarvaBhashika: Translating Tulu to diverse languages
**SarvaBhashika** is an interactive tool that allows users to **draw Tulu characters** and get translations in multiple languages, including **Kannada**, **English**, **Malayalam**, and **Hindi**. This project utilizes deep learning models for character recognition and Streamlit for providing a user-friendly web interface.

The tool promotes the Tulu language by helping users translate handwritten Tulu characters into these languages in real-time.

## Features

- **Draw Tulu Characters**: Users can draw Tulu characters directly on the canvas, and the system will recognize the character.
- **Multi-Language Translation**: Translates the recognized Tulu character into **Kannada**, **English**, **Malayalam**, and **Hindi**.
- **Real-Time Feedback**: Instant translation of the drawn character.
- **Streamlit Interface**: A simple, interactive interface built with Streamlit, making it easy for users to interact with the system.

## System Architecture

- **Front-End**: Built using **Streamlit**, which allows users to draw characters on a canvas and see the translation instantly.
- **Back-End**: A deep learning model (using **TensorFlow** or **PyTorch**) for recognizing Tulu characters and providing translations.
- **Translation**: The recognized character is translated into **Kannada**, **English**, **Malayalam**, and **Hindi** based on pre-trained mappings.

## Prerequisites

### Software Requirements

- **Python 3.8+**
- **Streamlit**: For building the web-based interface
- **TensorFlow** or **PyTorch**: For machine learning model training and inference
- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **Pandas**: For handling data
- **Pillow**: For image handling in the drawing interface

### Hardware Requirements

- A system with at least **4GB RAM** and **2GB of free disk space**.
- **GPU support** (optional) for faster model inference but not required.

## Installation

### 1. Clone the Repository

Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/Varnamitra-Tulu-word-translation.git
```
### 2. Navigate to the Project Folder

```bash
cd Varnamitra-Tulu-word-translation
```
### 3. Install Dependencies

Create a virtual environment (optional but recommended) and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```
The requirements.txt file includes all the necessary Python libraries for running the project.
4. Pre-trained Model
Ensure that the trained model (e.g., model.h5 or equivalent) is saved in the /model/ directory or adjust the path in the code for loading the model. If the model is not included, you can retrain it using the dataset provided in the data/ directory.

### 5. Running the Application
To start the Streamlit application, run the following command:
```bash
streamlit run app.py
```
This will launch the Streamlit app in your default web browser. You can now draw Tulu characters on the canvas, and the app will recognize and translate the character into Kannada, English, Malayalam, and Hindi.
## Project Structure
```bash
Varnamitra-Tulu-word-translation/
├── model/
│   └── model.h5              # Trained machine learning model for Tulu character recognition and translation
├── data/
│   └── dataset/              # Dataset for training the model
│   └── character_mappings.csv # CSV file with Tulu to Kannada/English/Malayalam/Hindi character mappings
├── src/
│   ├── app.py                # Streamlit application for user interface
│   ├── translator.py         # Functions for translation and character recognition
│   ├── image_processing.py   # Image preprocessing utilities for drawing recognition
│   └── canvas.py             # Canvas functionality for drawing Tulu characters
├── requirements.txt          # List of required Python packages
└── README.md                 # Project overview and documentation
```
### How to Use
1. Launch the Streamlit App: Follow the installation steps above and run the app.
2. Draw a Tulu Character: Use the drawing canvas to draw a Tulu character.
3. View the Translation: Once the character is drawn, click the Recognize button, and the app will display the translations in Kannada, English, Malayalam, and Hindi.
## Contact
For issues or inquiries, please create an issue on the GitHub repository or contact the maintainer:
#### Email: deekshitha0825@gmail.com
#### GitHub: https://github.com/dee2003
Thank you for your interest in Varnamitra! We hope this project helps in preserving and promoting the Tulu language by providing a simple and intuitive way to translate Tulu characters.
