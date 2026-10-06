import pandas as pd


def recommend_songs(emotion, number_of_songs=5):
    songs = pd.read_csv("songs.csv")

    matching_songs = songs[
        songs["emotion"].str.lower() == emotion.lower()
    ]

    if matching_songs.empty:
        matching_songs = songs[
            songs["emotion"].str.lower() == "neutral"
        ]

    return matching_songs.head(number_of_songs)
