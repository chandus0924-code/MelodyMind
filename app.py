import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace
from recommender import recommend_songs


st.set_page_config(
    page_title="MelodyMind",
    page_icon="🎵",
    layout="centered"
)


st.title("MelodyMind")
st.subheader("Mood-Based Music Recommendation via Facial Expression")

st.write(
    "Take a photo using your camera. MelodyMind will detect your facial "
    "expression and recommend suitable songs."
)


photo = st.camera_input("Take a picture")


if photo is not None:

    image = Image.open(photo)
    image_array = np.array(image)

    st.image(image, caption="Your captured image", use_container_width=True)

    with st.spinner("Analyzing your facial expression..."):

        try:
            result = DeepFace.analyze(
                image_array,
                actions=["emotion"],
                enforce_detection=False
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]

            st.success(
                f"Detected emotion: {emotion.capitalize()}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            st.header("Recommended Songs")

            songs = recommend_songs(emotion, number_of_songs=5)

            for _, song in songs.iterrows():

                st.subheader(
                    f"{song['title']} - {song['artist']}"
                )

                st.write(f"Genre: {song['genre']}")

                st.markdown(
                    f"[Listen to this song]({song['url']})"
                )

                st.divider()

        except Exception as error:

            st.error(
                "Unable to detect the emotion. "
                "Please take a clear picture with your face visible."
            )

            st.write("Technical details:", error)
