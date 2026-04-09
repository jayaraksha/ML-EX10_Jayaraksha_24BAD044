print("Jayaraksha 24BAD044")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import seaborn as sns

ratings = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\ratings.csv")
movies = pd.read_csv(r"C:\Users\hp\Downloads\archive (7)\ml-latest-small\movies.csv")

ratings = ratings[['userId', 'movieId', 'rating']]
movies = movies[['movieId', 'title']]

df = pd.merge(ratings, movies, on='movieId')

user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

matrix = user_item_matrix.values

k = 20

nmf = NMF(n_components=k, init='nndsvd', random_state=42, max_iter=500)
W = nmf.fit_transform(matrix)
H = nmf.components_

matrix_reconstructed = np.dot(W, H)

predicted_df = pd.DataFrame(matrix_reconstructed, index=user_item_matrix.index, columns=user_item_matrix.columns)

mask = matrix != 0
actual = matrix[mask]
predicted = matrix_reconstructed[mask]

rmse = np.sqrt(mean_squared_error(actual, predicted))
print("RMSE:", rmse)

def precision_recall_at_k(pred_df, original_matrix, k=10, threshold=3.5):
    precisions = []
    recalls = []

    for user in pred_df.index:
        user_actual = original_matrix.loc[user]
        user_pred = pred_df.loc[user]

        relevant_items = set(user_actual[user_actual >= threshold].index)

        if len(relevant_items) == 0:
            continue

        top_k_items = set(user_pred.sort_values(ascending=False).head(k).index)

        true_positives = len(top_k_items & relevant_items)

        precision = true_positives / k
        recall = true_positives / len(relevant_items)

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    return avg_precision, avg_recall

precision_k, recall_k = precision_recall_at_k(predicted_df, user_item_matrix, k=10, threshold=3.5)

print("Precision@10:", precision_k)
print("Recall@10:", recall_k)

plt.figure(figsize=(8,6))
sns.heatmap(user_item_matrix.iloc[:10, :10], cmap='YlGnBu')
plt.title("Original User-Item Matrix (Sample)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(predicted_df.iloc[:10, :10], cmap='YlGnBu')
plt.title("Reconstructed Matrix using NMF")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.show()

plt.figure(figsize=(8,6))
plt.bar(range(k), W[0], label='User 1 Features')
plt.title("Latent Feature Representation (User)")
plt.xlabel("Latent Factors")
plt.ylabel("Feature Strength")
plt.legend()
plt.grid(True)
plt.show()

user_id = 1

if user_id in predicted_df.index:
    user_ratings = predicted_df.loc[user_id]
    original_rated = user_item_matrix.loc[user_id]

    recommendations = user_ratings[original_rated == 0].sort_values(ascending=False)

    top_n = recommendations.head(10)

    print("\nTop 10 Recommended Movies for User", user_id)
    print(top_n)

    plt.figure(figsize=(10,6))
    top_n.plot(kind='bar', label='Predicted Rating')
    plt.title("Top 10 Recommended Movies")
    plt.xlabel("Movies")
    plt.ylabel("Predicted Rating")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("User not found")
