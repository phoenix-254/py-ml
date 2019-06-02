import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

from os import path


class MusicRecommender:
    music_data_set_file_name = 'files/music.csv'
    trained_model_file_name = 'files/music-recommender.joblib'

    def get_genre(self, age, gender):
        if not self.is_trained_model_exists():
            self.train_the_model()

        # Get the Persisted model
        model = joblib.load(self.trained_model_file_name)

        # Predict the Genre
        predictions = model.predict([[age, gender]])

        return predictions[0]

    def is_trained_model_exists(self):
        return path.exists(self.trained_model_file_name)

    def train_the_model(self):
        # Importing the data
        music_data = pd.read_csv(self.music_data_set_file_name)

        # model input data set
        X = music_data.drop(columns=['genre'])

        # model output data set
        y = music_data['genre']

        # Split the data into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # Create a model
        model = DecisionTreeClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Persist the model
        joblib.dump(model, self.trained_model_file_name)
