print("jayaraksha 24BAD044")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

ratings = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\ratings.csv")
movies = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\movies.csv")

ratings = ratings[['userId', 'movieId', 'rating']]
movies = movies[['movieId', 'title']]

df = pd.merge(ratings, movies, on='movieId')

print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

user_item_matrix = train_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

matrix = user_item_matrix.values
users = user_item_matrix.index
movies_ids = user_item_matrix.columns

user_ratings_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(matrix_demeaned, k=50)
sigma = np.diag(sigma)

predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(predicted_ratings, columns=movies_ids, index=users)

actual = []
predicted = []

for row in test_df.itertuples():
    user = row.userId
    movie = row.movieId
    rating = row.rating

    if user in preds_df.index and movie in preds_df.columns:
        pred_rating = preds_df.loc[user, movie]
        actual.append(rating)
        predicted.append(pred_rating)

rmse_value = np.sqrt(mean_squared_error(actual, predicted))
mae_value = mean_absolute_error(actual, predicted)

print("\nRMSE:", rmse_value)
print("MAE:", mae_value)

actual_ratings = actual[:100]
predicted_ratings_plot = predicted[:100]

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(actual_ratings) + 1), actual_ratings, marker='o', label='Actual Ratings')
plt.plot(range(1, len(predicted_ratings_plot) + 1), predicted_ratings_plot, marker='x', label='Predicted Ratings')
plt.title("Actual vs Predicted Ratings")
plt.xlabel("Test Instances")
plt.ylabel("Rating")
plt.legend()
plt.grid(True)
plt.show()

factor_values = [20, 40, 60, 80, 100]
rmse_scores = []

for k in factor_values:
    if k < min(matrix_demeaned.shape):
        U, sigma, Vt = svds(matrix_demeaned, k=k)
        sigma = np.diag(sigma)
        temp_pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        temp_preds_df = pd.DataFrame(temp_pred, columns=movies_ids, index=users)

        temp_actual = []
        temp_predicted = []

        for row in test_df.itertuples():
            user = row.userId
            movie = row.movieId
            rating = row.rating

            if user in temp_preds_df.index and movie in temp_preds_df.columns:
                temp_actual.append(rating)
                temp_predicted.append(temp_preds_df.loc[user, movie])

        temp_rmse = np.sqrt(mean_squared_error(temp_actual, temp_predicted))
        rmse_scores.append(temp_rmse)
    else:
        rmse_scores.append(None)

plt.figure(figsize=(8, 6))
plt.plot(factor_values, rmse_scores, marker='o', label='RMSE')
plt.title("RMSE vs Number of Latent Factors")
plt.xlabel("Number of Latent Factors")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

user_id = 1

if user_id in preds_df.index:
    rated_movie_ids = df[df['userId'] == user_id]['movieId'].unique()
    all_movie_ids = movies['movieId'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids and movie_id in preds_df.columns]

    recommendations = []
    for movie_id in unrated_movie_ids:
        pred_rating = preds_df.loc[user_id, movie_id]
        recommendations.append((movie_id, pred_rating))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    recommended_movies = pd.DataFrame(recommendations, columns=['movieId', 'Predicted Rating'])
    recommended_movies = recommended_movies.merge(movies, on='movieId')

    print("\nTop 10 Recommended Movies for User", user_id)
    print(recommended_movies[['title', 'Predicted Rating']])

    plt.figure(figsize=(10, 6))
    plt.bar(recommended_movies['title'], recommended_movies['Predicted Rating'], label='Predicted Rating')
    plt.title("Top 10 Recommended Movies")
    plt.xlabel("Movie Title")
    plt.ylabel("Predicted Rating")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print(f"User {user_id} not found in training data.")
