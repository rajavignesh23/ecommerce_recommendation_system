import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

products = pd.read_csv("testdata.csv")
user_history = pd.read_csv("user_history1.csv")

# Preprocessing
products['title'] = products['title'].astype(str)
products['tag'] = (products['title'] + ' ' + products['category_name'] + ' ' +
                    products['stars'].astype(str) + ' stars ' + 
                    products['price'].astype(str) + ' price ' + 
                    products['listPrice'].astype(str) + ' listPrice ' +
                    products['isBestSeller'].astype(str) + ' bestSeller ' +
                    products['boughtInLastMonth'].astype(str))

porter_stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([porter_stemmer.stem(word) for word in word_tokenize(text)])

products['tag'] = products['tag'].apply(stem_text)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vec_sparse = vectorizer.fit_transform(products['tag'])
svd = TruncatedSVD(n_components=100)
vec_reduced = svd.fit_transform(vec_sparse)
vec_reduced = np.ascontiguousarray(vec_reduced, dtype=np.float32)

index = faiss.IndexFlatL2(vec_reduced.shape[1])
faiss.normalize_L2(vec_reduced)
index.add(vec_reduced)

def recommend(user_history_df):
    user_products = products[products['title'].isin(user_history_df['title'])]
    user_indices = user_products.index.tolist()
    user_vecs = vec_reduced[user_indices]
    distances, indices = index.search(user_vecs, 6)
    
    recommendations = {}
    for i, category in enumerate(user_products['category_name']):
        top_product_indices = indices[i][1:]  
        category_products = products.iloc[top_product_indices]
        category_products = category_products[['title', 'stars', 'reviews', 'price']]
        recommendations[category] = category_products.values.tolist()
    
    return recommendations

recommendations = recommend(user_history)

print("Deals related to your views:")
for category, products in recommendations.items():
    print(f"\nCategory: {category}")
    if products:
        for product in products:
            name = product[0]
            stars = int(product[1]) if isinstance(product[1], float) else product[1]
            reviews = int(product[2]) if isinstance(product[2], float) else product[2]
            price = product[3]
            print(f"\nProduct: {name}")
            print(f"Rating: {'*' * stars} ({stars})")
            print(f"Reviews: ({reviews})")
            print(f"Price: ${price}")
    else:
        print("No recommendations in this category.")

plt.figure(figsize=(8, 6))
products['category_name'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Product Category Distribution')
plt.ylabel('')
plt.show()

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(vec_reduced)
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='YlGnBu')
plt.title('Product Similarity Heatmap')
plt.show()


avg_ratings = products.groupby('category_name')['stars'].mean().sort_values()
plt.figure(figsize=(10, 6))
avg_ratings.plot(kind='bar', color='skyblue')
plt.title('Average Product Rating per Category')
plt.xlabel('Category')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()

y_true = np.random.randint(0, 2, size=len(user_history))  # Example ground truth
y_pred = np.random.randint(0, 2, size=len(user_history))  # Example predictions

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

