from flask import Flask, request, jsonify
import joblib
from MNB import MultinomialNB

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/mnb_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Transform the text using the vectorizer
    text_vec = vectorizer.transform([text]).toarray()

    # Make the prediction
    prediction = model.predict(text_vec)
    probabilities = model.predict_proba(text_vec)[0]

    # Prepare the response
    sentiment = 'Positive' if prediction[0] == 0 else 'Negative'
    response = {
        'prediction': sentiment,
        'probabilities': {
            'Positive': probabilities[0],
            'Negative': probabilities[1]
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
