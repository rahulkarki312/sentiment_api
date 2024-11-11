from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from langdetect import detect
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata as un
from collections import OrderedDict

# Flask app initialization
app = Flask(__name__)
app.json.sort_keys = False
CORS(app)  # Enable CORS

# Load the models and vectorizers
english_model = joblib.load('models/mnb_model.pkl')
english_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

nepali_model = joblib.load('models/nepali/mnb_model.pkl')
nepali_vectorizer = joblib.load('models/nepali/tfidf_vectorizer.pkl')


# Helper function to detect if the text is Nepali
def is_nepali(text):
    nepali_char_pattern = re.compile("[\u0900-\u097F]+")  # Unicode range for Nepali characters
    return bool(nepali_char_pattern.search(text))


# English Preprocessing function
def preprocess_text(text):
    steps =  OrderedDict()

    # Step 1: Lowercasing
    text_lower = text.lower()
    steps['lowercase'] = text_lower

    # Step 2: Removing special characters and punctuation
    text_cleaned = re.sub(r'[^\w\s]', '', text_lower)
    steps['special_characters_removed'] = text_cleaned

    # Step 3: Tokenization
    tokens = word_tokenize(text_cleaned)
    steps['tokens'] = tokens

    negation_words = {"not", "no", "never", "none"}
    processed_tokens = tokens

    # ignoring negation handling logic for displaying only major steps
    # Step 4: Negation handling and token processing
    # processed_tokens = []
    # negation_flag = False
    # negation_word = None
    # negation_words = {"not", "no", "never", "none"}
    # insignificant_words = set()  # Define any insignificant words here if needed
    #
    # for i, token in enumerate(tokens):
    #     if negation_flag:
    #         if token not in insignificant_words:
    #             processed_tokens.append(f"{negation_word}_{token}")
    #             negation_flag = False
    #     elif token in negation_words:
    #         negation_flag = True
    #         negation_word = token
    #     else:
    #         processed_tokens.append(token)
    # steps['negation_handled'] = processed_tokens

    # Step 5: Removing stopwords
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words - negation_words
    tokens_no_stopwords = [word for word in processed_tokens if word not in stop_words]
    steps['stopwords_removed'] = tokens_no_stopwords

    # Step 6: Stemming
    stemmer = nltk.PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_no_stopwords]
    steps['stemmed'] = tokens_stemmed

    # Step 7: Removing short words
    tokens_filtered = [word for word in tokens_stemmed if len(word) > 1]
    steps['short_words_removed'] = tokens_filtered

    # Step 8: Reconstructing text
    cleaned_text = ' '.join(tokens_filtered)
    steps['final_reconstructed_text'] = cleaned_text

    return steps


# Nepali Preprocessing function
def preprocess_nepali_text(text):
    steps = OrderedDict()

    # Negation, significant, and insignificant words
    negation_words = {"छैन", "होइन", "न"}
    significant_words = {'नराम्रो', 'बिरुद्ध', 'राम्रो', 'ठीक', 'छैन', 'थिएन', 'भएन'}
    insignificant_words = {'यो' ,'त्यो', 'कोरोना', 'कोभीड', 'कोभिड', 'कोभडि', 'भाइरस',  'सरकार', 'नेपाल',  'संक्रमण', 'आज', 'माहामारी', 'महामारी' ,'खोप', 'लकडाउन', 'काठ्माडौँ'}
    stop_words = set(stopwords.words('nepali'))
    stop_words = stop_words - significant_words

    def nepali_stemmer(word):
        suffixes = ['हरू', 'मा', 'ियो', 'को', 'ले']
        for suffix in suffixes:
            if word.endswith(suffix):
                return word[:-len(suffix)]
        return word

    # Step 1: Removing special characters
    text_cleaned = re.sub(r'[!?,=|।+\-._\'\";:{}()\[\]<>/@#$%^&*~]', '', text)
    steps['special_characters_removed'] = text_cleaned


    # Step 2: Removing unnecessary punctuation
    #text_no_punctuation = re.sub(r'[!|।]+', '', text_cleaned)
    #steps['punctuation_removed'] = text_no_punctuation

    # Step 3: Exclude Nepali numbers
    text_no_numbers = re.sub(r'[१२३४५६७८९०]', '', text_cleaned)
    steps['nepali_numbers_removed'] = text_no_numbers

    # Step 4: Tokenization
    tokens = word_tokenize(text_no_numbers)
    steps['tokens'] = tokens

    # Step 5: Removing stopwords
    tokens_no_stopwords = [word for word in tokens if word not in stop_words]
    steps['stopwords_removed'] = tokens_no_stopwords

    # Step 6: Removing insignificant words
    tokens_significant = [word for word in tokens_no_stopwords if word not in insignificant_words]
    steps['insignificant_words_removed'] = tokens_significant

    # Step 7: Stemming
    tokens_stemmed = [nepali_stemmer(word) for word in tokens_significant]
    steps['stemmed'] = tokens_stemmed

    # Step 8: Normalize Unicode
    normalized_text = ' '.join(tokens_stemmed)
    normalized_text = un.normalize('NFC', normalized_text)
    steps['unicode_normalized'] = normalized_text

    return steps


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    steps = {}

    if is_nepali(text):
        # Preprocess Nepali text and get the steps
        preprocessing_steps = preprocess_nepali_text(text)
        preprocessed_text = preprocessing_steps['unicode_normalized']
        text_vec = nepali_vectorizer.transform([preprocessed_text]).toarray()
        prediction = nepali_model.predict(text_vec)
        probabilities = nepali_model.predict_proba(text_vec)[0]
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    else:
        # Preprocess English text and get the steps
        preprocessing_steps = preprocess_text(text)
        preprocessed_text = preprocessing_steps['final_reconstructed_text']
        text_vec = english_vectorizer.transform([preprocessed_text]).toarray()
        prediction = english_model.predict(text_vec)
        probabilities = english_model.predict_proba(text_vec)[0]
        sentiment = 'Positive' if prediction[0] == 0 else 'Negative'

    # Prepare the response
    response = {
        'review': text,
        'preprocessing_steps': preprocessing_steps,
        'prediction': sentiment,
        'probabilities': {
            'Positive': probabilities[1] if is_nepali(text) else probabilities[0],
            'Negative': probabilities[0] if is_nepali(text) else probabilities[1],
        },
        'language': 'Nepali' if is_nepali(text) else 'English'
    }

    return jsonify(response)


# to run locally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
