from music_recommender import MusicRecommender

test_data = [[21, 1], [22, 0]]

mr = MusicRecommender()

for data in test_data:
    age = data[0]
    gender = data[1]

    genre = mr.get_genre(age, gender)

    print(f'A Person having Age: {age} and Gender: {gender} '
          f'is predicted to like {genre} Genre of music')
