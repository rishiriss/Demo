from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('bosch_item_based_collaborative_filtering.csv')

# Normalize 'Avg. Rating' and 'Co-Purchase Count'
df['Avg. Rating'] = (df['Avg. Rating'] - df['Avg. Rating'].min()) / (df['Avg. Rating'].max() - df['Avg. Rating'].min())
df['Co-Purchase Count'] = (df['Co-Purchase Count'] - df['Co-Purchase Count'].min()) / (df['Co-Purchase Count'].max() - df['Co-Purchase Count'].min())

# Create item-feature matrix
item_features = df[['Avg. Rating', 'Co-Purchase Count']].values

# Compute the cosine similarity matrix
item_similarity = cosine_similarity(item_features)
item_similarity_df = pd.DataFrame(item_similarity, index=df['Product ID'], columns=df['Product ID'])

# Recommendation function
def recommend_next_best_product(product_id, top_n=5):
    if product_id in item_similarity_df.index:
        similar_products = item_similarity_df[product_id].sort_values(ascending=False).index[1:top_n+1]
        recommendations = df[df['Product ID'].isin(similar_products)][['Product ID', 'Product Name', 'Avg. Rating']]
        return recommendations.to_dict(orient='records')
    else:
        return None

# Route to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Route to get recommendations based on product ID
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    product_id = int(data.get('product_id'))
    recommendations = recommend_next_best_product(product_id)
    
    if recommendations:
        return jsonify(recommendations)
    else:
        return jsonify({'error': 'Product ID not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
