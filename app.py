from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data/recipes.csv")

# Fill NaN values in relevant columns
data['RecipeIngredientQuantities'] = data['RecipeIngredientQuantities'].fillna("")
data['RecipeIngredientParts'] = data['RecipeIngredientParts'].fillna("")
data['CookTime'] = data['CookTime'].fillna("N/A")
data['PrepTime'] = data['PrepTime'].fillna("N/A")
data['TotalTime'] = data['TotalTime'].fillna("N/A")
data['RecipeCategory'] = data['RecipeCategory'].fillna("N/A")
data['RecipeServings'] = data['RecipeServings'].fillna("N/A")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['RecipeIngredientParts'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['Calories', 'CarbohydrateContent', 'ProteinContent']])

# Combine Features for KNN
X_combined = hstack([X_numerical, X_ingredients])

# Train KNN Model on Combined Features
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_combined)

# Recommendation function
def recommend_recipes(input_features):
    # Normalize and prepare input features
    input_numerical_scaled = scaler.transform([input_features[:3]])
    input_ingredients_transformed = vectorizer.transform([input_features[3]])
    input_combined = hstack([input_numerical_scaled, input_ingredients_transformed])

    # Find nearest neighbors
    distances, indices = knn.kneighbors(input_combined)

    # Fetch and return the recommended recipes
    recommendations = data.iloc[indices[0]]
    recommendations['AllIngredients'] = recommendations['RecipeIngredientParts'].apply(
        lambda x: ', '.join(x.split(',')) if isinstance(x, str) else x
    )
    return recommendations[['Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts',
                            'CookTime', 'PrepTime', 'TotalTime', 'RecipeInstructions',
                            'RecipeCategory', 'RecipeServings', 'Images', 'AllIngredients']]

# Function to truncate text
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

# Route to render homepage and handle POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect user input
            calories = float(request.form['calories'])
            carbohydrates = float(request.form['carbohydrates'])
            protein = float(request.form['protein'])
            ingredients = request.form['ingredients']

            # Prepare the input features
            input_features = [calories, carbohydrates, protein, ingredients]

            # Get recommendations
            recommendations = recommend_recipes(input_features)

            # Render results on the webpage
            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),
                                   truncate=truncate)
        except ValueError:
            # Handle invalid input
            return render_template('index.html', recommendations=[],
                                   error="Invalid input. Please enter valid numbers for nutritional values.")

    # Render the homepage with no recommendations (initial load)
    return render_template('index.html', recommendations=[])

# Route to show recipe details
@app.route('/recipe/<int:recipe_id>')
def recipe_details(recipe_id):
    recipe = data.iloc[recipe_id]
    return render_template('recipe_details.html', recipe=recipe)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
