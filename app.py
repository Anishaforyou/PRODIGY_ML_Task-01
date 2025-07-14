from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # MUST be POST
def predict():
    if request.method == 'POST':
        try:
            sqft = float(request.form['sqft'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])

            features = np.array([[sqft, bedrooms, bathrooms]])
            prediction = model.predict(features)[0]
            price = f"${prediction:,.2f}"

            return render_template('index.html', prediction_text=f"Predicted House Price: {price}")
        except Exception as e:
            return render_template('index.html', prediction_text="❌ Invalid input. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
