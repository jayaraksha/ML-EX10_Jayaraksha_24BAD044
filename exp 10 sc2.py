print("24bad044 jayaraksha")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

k = 20
nmf_model = NMF(n_components=k, init='random', random_state=42, max_iter=500)

W = nmf_model.fit_transform(user_item_matrix.values)
H = nmf_model.components_

print("\nUser-Feature Matrix Shape:", W.shape)
print("Item-Feature Matrix Shape:", H.shape)

reconstructed_matrix = np.dot(W, H)
preds_df = pd.DataFrame(reconstructed_matrix, columns=user_item_matrix.columns, index=user_item_matrix.index)

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

def precision_recall_at_k(actual_matrix, predicted_matrix, k=10, threshold=3.5):
    precisions = []
    recalls = []

    for user in range(actual_matrix.shape[0]):
        actual_ratings = actual_matrix[user]
        predicted_ratings = predicted_matrix[user]

        relevant_items = set(np.where(actual_ratings >= threshold)[0])
        recommended_items = np.argsort(predicted_ratings)[::-1][:k]
        recommended_set = set(recommended_items)

        true_positives = len(relevant_items & recommended_set)

        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

precision_k, recall_k = precision_recall_at_k(actual, predicted, k=10)

print("\nPrecision@10:", precision_k)
print("Recall@10   :", recall_k)

def recommend_movies_nmf(user_id, preds_df, original_ratings, movies_df, num_recommendations=10):
    sorted_user_predictions = preds_df.loc[user_id].sort_values(ascending=False)

    user_data = original_ratings[original_ratings.userId == user_id]
    user_watched_movies = user_data["movieId"].tolist()

    recommendations = pd.DataFrame(sorted_user_predictions).reset_index()
    recommendations.columns = ["movieId", "PredictedRating"]

    recommendations = recommendations[~recommendations["movieId"].isin(user_watched_movies)]
    recommendations = recommendations.merge(movies_df, on="movieId")
    recommendations = recommendations.sort_values("PredictedRating", ascending=False).head(num_recommendations)

    return recommendations

recommendations = recommend_movies_nmf(1, preds_df, ratings, movies, 10)

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
plt.title("Reconstructed User-Item Matrix using NMF (Sample)")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(recommendations['title'], recommendations['PredictedRating'])
plt.xlabel("Predicted Rating")
plt.ylabel("Movie Title")
plt.title("Top 10 Recommended Movies for User 1 (NMF)")
plt.gca().invert_yaxis()
plt.show()

k_values = [5, 10, 15, 20, 25, 30, 40]
rmse_list = []

for k in k_values:
    nmf_temp = NMF(n_components=k, init='random', random_state=42, max_iter=500)
    W_temp = nmf_temp.fit_transform(user_item_matrix.values)
    H_temp = nmf_temp.components_
    preds_temp = np.dot(W_temp, H_temp)

    rmse_k = np.sqrt(mean_squared_error(actual[mask], preds_temp[mask]))
    rmse_list.append(rmse_k)

plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_list, marker='o')
plt.title("Error vs Number of Latent Factors (k) - NMF")
plt.xlabel("Number of Latent Factors (k)")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

print("\nK Values and RMSE:")
for i in range(len(k_values)):
    print(f"k = {k_values[i]} --> RMSE = {rmse_list[i]:.4f}")
