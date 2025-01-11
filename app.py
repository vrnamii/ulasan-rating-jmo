import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load stopwords and stemmer
stop_words = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function to preprocess reviews
def preprocess_review(review):
    review = re.sub(r'https?://\S+|www\.\S+', '', review)
    review = re.sub(r'<.*?>', '', review)
    review = re.sub(r"[" 
                    u"\U0001F600-\U0001F64F" 
                    u"\U0001F300-\U0001F5FF" 
                    u"\U0001F680-\U0001F6FF" 
                    u"\U0001F1E0-\U0001F1FF" 
                    "]+", '', review, flags=re.UNICODE)
    review = re.sub(r'[0-9]+', '', review)
    review = re.sub(r'\$\w*', '', review)
    review = re.sub(r'^RT[\s]+', '', review)
    review = re.sub(r'#', '', review)
    translator = str.maketrans('', '', string.punctuation)
    review = review.translate(translator)
    review = review.lower()
    tokens = word_tokenize(review)
    with open("slangwords.txt") as f:
        slang_dict = eval(f.read())
    pattern = re.compile(r'\b(' + '|'.join(slang_dict.keys()) + r')\b')
    tokens = [pattern.sub(lambda x: slang_dict[x.group()], word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load TF-IDF vectorizer and SVM model
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))

# Streamlit app
st.title("Klasifikasi Rating Ulasan Aplikasi JMO BPJS Ketenagakerjaan")
st.write("""
Aplikasi ini menggunakan metode **TF-IDF** untuk pemrosesan teks dan **SVM** untuk klasifikasi 
rating ulasan pengguna pada aplikasi **JMO BPJS Ketenagakerjaan**.
""")

# Initialize session state to store results
if "results" not in st.session_state:
    st.session_state["results"] = {}

# Multiple file input
uploaded_files = st.file_uploader("Unggah File CSV (kolom: 'Content')", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        ulasan_data = pd.read_csv(uploaded_file)

        # Check if 'Content' column exists
        if 'Content' in ulasan_data.columns:
            st.write(f"Data dari file: {file_name}")
            st.write(ulasan_data)

            if st.button(f"Klasifikasi Ulasan: {file_name}"):
                # Process reviews and classify
                processed_reviews = []
                predicted_ratings = []
                
                for _, row in ulasan_data.iterrows():
                    processed_review = preprocess_review(row['Content'])
                    vectorized_review = tfidf_vectorizer.transform([processed_review]).toarray()  # Convert to dense
                    rating_prediction = svm_model.predict(vectorized_review)[0]
                    
                    processed_reviews.append(processed_review)
                    predicted_ratings.append(rating_prediction)
                
                # Add results to DataFrame
                ulasan_data['Processed Review'] = processed_reviews
                ulasan_data['Predicted Rating'] = predicted_ratings

                # Store results in session state
                st.session_state["results"][file_name] = ulasan_data

        else:
            st.error(f"File CSV: {file_name} tidak memiliki kolom 'Content'. Harap pastikan format file sesuai.")

# Display all classification results stored in session state
if st.session_state["results"]:
    st.write("### Hasil Klasifikasi Semua File:")
    for file_name, result_df in st.session_state["results"].items():
        st.write(f"**Hasil Klasifikasi untuk File: {file_name}**")
        st.write(result_df)
