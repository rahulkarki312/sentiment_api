import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata as un

# Example list of Nepali negation words (expand as needed)

negation_words = {"छैन", "होइन", "न"}


significant_words = {'नराम्रो','बिरुद्ध', 'राम्रो','ठीक','छैन', 'थिएन', 'भएन'}
insignificant_words = {'यो' ,'त्यो', 'कोरोना', 'कोभीड', 'कोभिड', 'कोभडि', 'भाइरस',  'सरकार', 'नेपाल',  'संक्रमण', 'आज', 'माहामारी', 'महामारी' ,'खोप', 'लकडाउन', 'काठ्माडौँ'}

# Load Nepali stopwords

stop_words = set(stopwords.words('nepali'))
stop_words = stop_words - significant_words
#print(stop_words)

def nepali_stemmer(word):
    # Common Nepali suffixes
    suffixes = ['हरू','हरु','मा','िसकेको','िएको','को','ले','लाई','्न','्ने','ने','्दै','िन्छ','्छ','्थे','्थेँ','्थी','्थ्यौं','्यो','्नु','्यौँ', '्छौँ']

    for suffix in suffixes:
        if word.endswith(suffix):
            # Remove the suffix without additional modification
            return word[:-len(suffix)]

    return word



def preprocess_nepali_text(text):
    # Remove special characters like '!', ',', '.', '-'
    text = re.sub(r'[!?,=|।+\-]', '', text)

    # Remove unnecessary punctuation like '!', ',', '।' at the end of sentences, but keep the context words
    text = re.sub(r'[!|।]+', '', text)  #

    # Exclude Nepali numbers (१२३४५६७८९०)
    text = re.sub(r'[१२३४५६७८९०]', '', text)

    # tokenize
    tokens = word_tokenize(text)

    processed_tokens = [word for word in tokens if word not in stop_words]

    processed_tokens = [word for word in tokens if word not in insignificant_words]

    processed_tokens = [nepali_stemmer(word) for word in processed_tokens]

    text = ' '.join(processed_tokens)

    # Normalize Unicode
    text = un.normalize('NFC', text)

    # Additional preprocessing can be added here

    return text.strip()  # Remove leading/trailing whitespace

text = 'गुणस्तर हेर्दा जुत्ता एकदमै- - राम्रो! ! छ, वाउ |!'
print(preprocess_nepali_text(text))