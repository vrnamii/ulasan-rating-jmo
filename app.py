import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import pickle

# Pastikan semua dataset NLTK diunduh
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords dan stemmer
stop_words = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk preprocessing teks
def preprocess_review(review):
    # Hilangkan URL, HTML, dan emoji
    review = re.sub(r'https?://\S+|www\.\S+', '', review)
    review = re.sub(r'<.*?>', '', review)
    review = re.sub("[" 
                    u"\U0001F600-\U0001F64F" 
                    u"\U0001F300-\U0001F5FF" 
                    u"\U0001F680-\U0001F6FF" 
                    u"\U0001F1E0-\U0001F1FF" 
                    "]+", '', review, flags=re.UNICODE)
    review = re.sub(r'[0-9]+', '', review)
    review = re.sub(r'\$\w*', '', review)
    review = re.sub(r'^RT[\s]+', '', review)
    review = re.sub(r'#', '', review)

    # Hilangkan tanda baca dan ubah ke lowercase
    translator = str.maketrans('', '', string.punctuation)
    review = review.translate(translator).lower()

    # Tokenisasi
    tokens = word_tokenize(review)

    # Normalisasi slang words
    with open("slangwords.txt") as f:
        slang_dict = eval(f.read())
    pattern = re.compile(r'\b(' + '|'.join(slang_dict.keys()) + r')\b')
    tokens = [pattern.sub(lambda x: slang_dict[x.group()], word) for word in tokens]

    # Hilangkan stopwords dan stem kata
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Load model dan vectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))

# Aplikasi Streamlit
st.title("Klasifikasi Rating Ulasan Aplikasi JMO BPJS Ketenagakerjaan")

# Input file CSV
uploaded_files = st.file_uploader("Upload File CSV", type=["csv"], accept_multiple_files=True)

# Simpan hasil klasifikasi
results = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Baca data
        df = pd.read_csv(uploaded_file)

        # Pastikan kolom 'Content' ada
        if 'Content' in df.columns:
            st.write(f"Data dari file: {uploaded_file.name}")
            st.write(df)

            # Tombol klasifikasi untuk masing-masing file
            if st.button(f'Klasifikasi Data dari {uploaded_file.name}'):
                processed_reviews = []
                predictions = []

                for index, row in df.iterrows():
                    # Preprocess ulasan
                    processed_review = preprocess_review(row['Content'])

                    # Vectorisasi dan prediksi
                    vectorized_review = tfidf_vectorizer.transform([processed_review])
                    prediction = svm_model.predict(vectorized_review)[0]

                    processed_reviews.append(processed_review)
                    predictions.append(prediction)

                # Tambahkan hasil ke dataframe
                df['Processed Review'] = processed_reviews
                df['Prediction'] = predictions

                # Simpan hasil klasifikasi
                results[uploaded_file.name] = df

                # Tampilkan hasil
                st.write(f"Hasil Klasifikasi dari {uploaded_file.name}:")
                st.write(df)

# Tampilkan hasil klasifikasi sebelumnya (jika ada)
if results:
    st.write("Hasil Klasifikasi Sebelumnya:")
    for file_name, result_df in results.items():
        st.write(f"Hasil dari {file_name}:")
        st.write(result_df)

