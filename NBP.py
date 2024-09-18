from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from google.colab.output import eval_js
from IPython.display import display, Javascript

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset (upload dataset via Colab interface)
df = pd.read_csv('/content/bosch_item_based_collaborative_filtering.csv')

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

# Route to render the homepage
@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Product Recommendation System</title>
        <script>
            async function getRecommendations() {
                const productId = document.getElementById('product_id').value;
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ product_id: productId })
                });
                
                if (response.ok) {
                    const recommendations = await response.json();
                    let output = "<h3>Recommended Products:</h3><ul>";
                    recommendations.forEach(product => {
                        output += `<li>${product['Product Name']} (Rating: ${product['Avg. Rating']})</li>`;
                    });
                    output += "</ul>";
                    document.getElementById('recommendations').innerHTML = output;
                } else {
                    document.getElementById('recommendations').innerHTML = "Product ID not found.";
                }
            }
        </script>
    </head>
    <body>
        <h1>Product Recommendation System</h1>
        <label for="product_id">Enter Product ID:</label>
        <input type="number" id="product_id" name="product_id">
        <button onclick="getRecommendations()">Get Recommendations</button>

        <div id="recommendations"></div>
    </body>
    </html>
    ''')

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

# Function to run the Flask app on Colab's local server
def run_app():
    # Display the app's URL within Colab
    display(Javascript('google.colab.kernel.proxyPort(5000, {"cache": true})'))
    app.run(host='0.0.0.0', port=5000)

# Call the function to run the app
run_app()
