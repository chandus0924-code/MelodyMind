# MelodyMind

MelodyMind is a mood-based music recommendation system that detects facial expressions and recommends songs according to the user's detected emotion.

## Features

- Captures an image using the camera
- Detects facial emotion using DeepFace
- Uses Streamlit for the user interface
- Recommends songs using a mood-based CSV dataset
- Provides YouTube links for recommended songs

## Technologies Used

- Python
- Streamlit
- OpenCV
- DeepFace
- Pandas
- TensorFlow
- CSV dataset

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
