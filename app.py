"""
Simple Flask web app to expose churn model through API
"""

from flask import Flask, jsonify, request, render_template
from deploy import load_model
import pandas as pd

app = Flask(__name__)

# Load the most recent trained model
model = load_model()


# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data as a JSON object
    input_data = request.json

    # Convert the input data to a Pandas DataFrame
    input_df = pd.DataFrame.from_dict(input_data, orient='index').transpose()

    # Make a prediction using the trained model
    prediction = model.predict_proba(input_df)[:,1]

    # Convert the prediction to a JSON object and return it
    return jsonify({'prediction': prediction.tolist()})

# Define the index page route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
