print("24bad044 jayaraksha")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds

ratings = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\ratings.csv")
movies = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\movies.csv")

print("Ratings Dataset:")
print(ratings.head())

print("\nMovies Dataset:")
print(movies.head())

print("\nMissing Values:")
print(ratings.isnull().sum())

user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

print("\nUser-Item Matrix Shape:", user_item_matrix.shape)
print(user_item_matrix.head())

user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
ratings_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

k = 50
U, sigma, Vt = svds(ratings_demeaned, k=k)
sigma = np.diag(sigma)

print("\nU Shape:", U.shape)
print("Sigma Shape:", sigma.shape)
print("Vt Shape:", Vt.shape)

predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

print("\nReconstructed Ratings Matrix:")
print(preds_df.head())

actual = user_item_matrix.values
predicted = preds_df.values
mask = actual > 0

rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
mae = mean_absolute_error(actual[mask], predicted[mask])

print("\nEvaluation Metrics:")
print("RMSE:", rmse)
print("MAE :", mae)

def recommend_movies(user_id, preds_df, original_ratings, movies_df, num_recommendations=10):
    sorted_user_predictions = preds_df.loc[user_id].sort_values(ascending=False)

    user_data = original_ratings[original_ratings.userId == user_id]
    user_watched_movies = user_data["movieId"].tolist()

    recommendations = pd.DataFrame(sorted_user_predictions).reset_index()
    recommendations.columns = ["movieId", "PredictedRating"]

    recommendations = recommendations[~recommendations["movieId"].isin(user_watched_movies)]
    recommendations = recommendations.merge(movies_df, on="movieId")
    recommendations = recommendations.sort_values("PredictedRating", ascending=False).head(num_recommendations)

    return recommendations

recommendations = recommend_movies(1, preds_df, ratings, movies, 10)

print("\nTop Recommended Movies for User 1:")
print(recommendations[["movieId", "title", "PredictedRating"]])

plt.figure(figsize=(10, 6))
sns.heatmap(user_item_matrix.iloc[:20, :20], cmap="YlGnBu")
plt.title("Original User-Item Matrix (Sample)")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(preds_df.iloc[:20, :20], cmap="YlOrRd")
plt.title("Reconstructed User-Item Matrix using SVD (Sample)")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(recommendations['title'], recommendations['PredictedRating'])
plt.xlabel("Predicted Rating")
plt.ylabel("Movie Title")
plt.title("Top 10 Recommended Movies for User 1 (SVD)")
plt.gca().invert_yaxis()
plt.show()

k_values = [10, 20, 30, 40, 50, 60, 80, 100]
rmse_list = []

for k in k_values:
    U, sigma, Vt = svds(ratings_demeaned, k=k)
    sigma = np.diag(sigma)
    preds = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    rmse_k = np.sqrt(mean_squared_error(actual[mask], preds[mask]))
    rmse_list.append(rmse_k)

plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_list, marker='o')
plt.title("Error vs Number of Latent Factors (k)")
plt.xlabel("Number of Latent Factors (k)")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

print("\nK Values and RMSE:")
for i in range(len(k_values)):
    print(f"k = {k_values[i]} --> RMSE = {rmse_list[i]:.4f}")
