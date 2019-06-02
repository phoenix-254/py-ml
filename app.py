import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing the clean data
music_data = pd.read_csv('files/music.csv')

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

# Make predictions
predictions = model.predict(X_test)

# Evaluate the accuracy
score = accuracy_score(y_test, predictions)

print(f'Accuracy Score: {score}')