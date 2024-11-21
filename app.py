import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

# --- Configure Streamlit ---
st.set_page_config(
    page_title="🎥 Movie Genre Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Add CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # Ensure `style.css` is in the same directory.

# --- Load Models and Vectorizers ---
nb_vectorizer = joblib.load("vectorizer.pkl")
lr_vectorizer = joblib.load("lr_vectorizer.pkl")

mapping_data = pd.read_csv("mapping.csv")
genre_mapping = mapping_data.set_index("id")["genre"].to_dict()
genre_count = len(genre_mapping)

# Load the training dataset for recommendations
training_data = pd.read_csv("train.csv")  # Ensure this file contains 'description', 'genres', 'movie_name'

# --- Prediction Functions ---
def predict_naive_bayes(description):
    predicted_genres = []
    X_input = nb_vectorizer.transform([description])

    for index in range(genre_count):
        nb_model = joblib.load(f"nb_models/nb_model_{index}.pkl")
        if nb_model.predict(X_input)[0] == 1:
            predicted_genres.append(index)

    return predicted_genres

def predict_logistic_regression(description):
    predicted_genres = []
    X_input = lr_vectorizer.transform([description])

    for index in range(genre_count):
        lr_model = joblib.load(f"lr_models/lr_model_{index}.pkl")
        if lr_model.predict(X_input)[0] == 1:
            predicted_genres.append(index)

    return predicted_genres

# --- Display Word Clouds ---
def display_wordclouds(predicted_genres):
    st.markdown('<div class="section-title">📊 Word Clouds</div>', unsafe_allow_html=True)
    cols = st.columns(3)

    for i, genre_id in enumerate(predicted_genres):
        genre_name = genre_mapping[genre_id]
        wordcloud_path = f"wordcloud/{genre_id}.png"
        
        if os.path.exists(wordcloud_path):
            with cols[i % 3]:
                st.image(wordcloud_path, caption=genre_name, use_container_width=True)

# --- Recommend Related Movies ---
def recommend_movies(predicted_genres):
    recommended_movies = []
    for genre_id in predicted_genres:
        genre_name = genre_mapping[genre_id]
        # Filter movies from training data that match the genre
        genre_movies = training_data[training_data['genre'].str.contains(genre_name, na=False)]
        recommended_movies.extend(genre_movies['movie_name'].tolist())

    # Remove duplicates and limit to 4-5 movies
    recommended_movies = list(set(recommended_movies))[:5]
    return recommended_movies

# --- Main Interface ---
st.markdown('<div class="full-page-title">🎥 Movie Genre Prediction App</div>', unsafe_allow_html=True)

if "scrolled" not in st.session_state:
    st.session_state.scrolled = False

if st.button("Get Started"):
    st.session_state.scrolled = True

if st.session_state.scrolled:
    st.markdown('<div class="section-title">📖 Enter Movie Description</div>', unsafe_allow_html=True)
    description = st.text_area("Enter a movie description:", "", height=150)

    if st.button("Predict Genres"):
        if description.strip():
            # Predictions
            nb_predictions = predict_naive_bayes(description)
            lr_predictions = predict_logistic_regression(description)

            # Display Predictions
            st.markdown('<div class="section-title">🎯 Predicted Genres</div>', unsafe_allow_html=True)
            st.write("**Naive Bayes Model:**")
            st.success(", ".join([genre_mapping[idx] for idx in nb_predictions]) if nb_predictions else "No genres predicted.")
            st.write("**Logistic Regression Model:**")
            st.success(", ".join([genre_mapping[idx] for idx in lr_predictions]) if lr_predictions else "No genres predicted.")

            # Word Clouds Section
            if lr_predictions:
                display_wordclouds(lr_predictions)

            # Recommendations Section
            if lr_predictions:
                st.markdown('<div class="section-title">🎥 Related Movies</div>', unsafe_allow_html=True)
                recommendations = recommend_movies(lr_predictions)
                if recommendations:
                    st.write("**Recommended Movies:**")
                    for movie in recommendations:
                        st.write(f"- {movie}")
                else:
                    st.write("No related movies found.")
        else:
            st.error("Please enter a valid movie description.")
