from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack  # Import sparse hstack to handle sparse matrix

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("recipe_final (1).csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(
    data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# Combine Features (Use sparse matrix to avoid memory issues)
X_combined = hstack([X_numerical, X_ingredients])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)


# Recipe recommendation function
def recommend_recipes(input_features):
    # Scale the numerical features
    input_features_scaled = scaler.transform([input_features[:7]])

    # Transform the ingredient list
    input_ingredients_transformed = vectorizer.transform([input_features[7]])

    # Combine scaled numerical features and transformed ingredient list
    input_combined = hstack([input_features_scaled, input_ingredients_transformed])

    # Get the nearest neighbors
    distances, indices = knn.kneighbors(input_combined)

    # Fetch and return the recommended recipes
    recommendations = data.iloc[indices[0]]
    return recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)


# Function to truncate text (if needed)
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


# Flask route to render the homepage and handle POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect and validate user input
            calories = float(request.form['calories'])
            fat = float(request.form['fat'])
            carbohydrates = float(request.form['carbohydrates'])
            protein = float(request.form['protein'])
            cholesterol = float(request.form['cholesterol'])
            sodium = float(request.form['sodium'])
            fiber = float(request.form['fiber'])
            ingredients = request.form['ingredients']

            # Prepare the input features
            input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]

            # Get the recommendations
            recommendations = recommend_recipes(input_features)

            # Render the results on the webpage
            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),
                                   truncate=truncate)
        except ValueError:
            # Handle cases where user input is not valid
            return render_template('index.html', recommendations=[],
                                   error="Invalid input. Please enter valid numbers for nutritional values.")

    # Render the homepage with no recommendations (initial load)
    return render_template('index.html', recommendations=[])


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
