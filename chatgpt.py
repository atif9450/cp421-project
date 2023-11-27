import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

# Load the ratings data
ratings = pd.read_csv('Ratings.csv', nrows=10000)  # Assuming your ratings data is in a file named 'ratings.csv'

# Load book metadata
books = pd.read_csv('Books.csv', nrows=10000)  # Assuming your book metadata is in a file named 'books.csv'
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)


# Merge ratings with book metadata
book_ratings = pd.merge(ratings, books, on='ISBN')

# Collaborative Filtering - User-Item Matrix
user_item_matrix = ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')

# Fill missing values with 0 (assuming no rating means the user hasn't read the book)
user_item_matrix = user_item_matrix.fillna(0)

# Check if the user has rated any books
if user_item_matrix.shape[0] == 0:
    print("User has not rated any books.")
    # Handle this case as needed (e.g., provide default recommendations)

# Calculate pairwise distances between items (books)
item_similarity = 1 - linear_kernel(user_item_matrix.T, user_item_matrix.T)

# Normalize the similarity values to a 0-1 scale
item_similarity_norm = MinMaxScaler().fit_transform(item_similarity)

# Content-Based Filtering - TF-IDF Vectorization
content_features = books['Book-Author'] + ' ' + books['Publisher'] + ' ' + books['Year-Of-Publication'].astype(str)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
content_matrix = tfidf_vectorizer.fit_transform(content_features)

# User Input (Example: User with ID 123 has rated a book)
user_input = pd.DataFrame({'User-ID': [276747], 'ISBN': ['0446605239'], 'Book-Rating': [10]})

# Content-Based Recommendation
input_book_index = books[books['ISBN'] == user_input['ISBN'].iloc[0]].index
if len(input_book_index) == 0:
    print(f"Book with ISBN {user_input['ISBN'].iloc[0]} not found.")
    # Handle this case as needed (e.g., provide default recommendations)
else:
    input_book_index = input_book_index[0]
    content_similarity = linear_kernel(content_matrix[input_book_index], content_matrix).flatten()
    content_recommendations = content_similarity.argsort()[:-11:-1]

    # Collaborative Filtering Recommendation
    collaborative_recommendations = item_similarity_norm[input_book_index].argsort()[:-11:-1]

    # Hybrid Recommendation (Simple Weighted Combination)
    alpha = 0.7  # Weight for collaborative recommendations
    beta = 0.3   # Weight for content-based recommendations
    hybrid_recommendations = alpha * collaborative_recommendations + beta * content_recommendations

    # Get Top-N Recommendations
    top_n_recommendations = hybrid_recommendations[:10]
    top_n_recommendations = top_n_recommendations.astype(int)
    top_n_isbns = user_item_matrix.columns[top_n_recommendations].tolist()
    top_n_books = books[books['ISBN'].isin(top_n_isbns)][['ISBN', 'Book-Title']]


    # Print Top Recommendations
    print("Top Recommendations:")
    print(top_n_books)
