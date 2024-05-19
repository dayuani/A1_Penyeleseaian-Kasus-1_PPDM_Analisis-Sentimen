import streamlit as st
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import altair as alt

st.set_page_config(layout="wide")


if "df" not in st.session_state:
	st.session_state.df = pd.DataFrame()
if "df_final" not in st.session_state:
    st.session_state.df_final = pd.DataFrame()
if "klasifikasi" not in st.session_state:
    st.session_state.klasifikasi = pd.DataFrame()

# Fungsi untuk mengupload dan menampilkan file CSV
def upload_and_display_csv():
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file, encoding='unicode_escape')
        st.write("Dataframe:")
        st.dataframe(st.session_state.df, width=3000, height=800)
        return None
    return None

# Fungsi untuk melakukan preprocessing data
def preprocess_data(df):
    if df is not None:
        st.write("Dataframe sebelum preprocessing:")
        st.dataframe(df)
        
        # Pastikan kolom 'content' ada dalam dataframe
        if 'content' in df.columns:
            with st.spinner("Sedang melakukan preprocess..."):
                # Filtering
                df_filtered = df[df['content'] != '']

                # Cleaning content
                def clean_content(text):
                    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
                    text = re.sub(r'#\w+', '', text)
                    text = re.sub(r'RT[\s]+', '', text)
                    text = re.sub(r'https?://\S+', '', text)
                    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
                    return text
                df_filtered['content'] = df_filtered['content'].apply(clean_content)

                # Case folding
                df_fold = df_filtered.copy()
                df_fold['content'] = df_fold['content'].apply(lambda x: x.lower())

                # Tokenisasi
                df_token = df_fold.copy()
                df_token['content'] = df_token['content'].apply(lambda x: x.split())

                #Normalisasi
                import json
                df_normal = df_token.copy()
                def read_dictionary_from_file(file_path):
                    with open(file_path, 'r') as file:
                        dictionary = json.load(file)
                    return dictionary
                
                file_path = 'slangwords.txt'  # Path to the JSON file
                my_dictionary = read_dictionary_from_file(file_path)
                
                def change(tokens):
                    new_list = []
                    for token in tokens:
                        if token in my_dictionary:
                            new_list.append(my_dictionary[token])
                        else:
                            new_list.append(token)
                    return new_list
                df_normal['content'] = df_normal['content'].apply(change)
                
                #Stopword Removal
                df_sw = df_normal.copy()
                def import_words_from_file(file_path):
                    word_list = []
                    with open(file_path, 'r') as file:
                        for line in file:
                            word_list.append(line.strip())
                    return word_list

                file_path = 'stopwords.txt'  # Nama file stopwords
                stopwords = import_words_from_file(file_path)

                def stopword_removal(tokens):
                    new_list = []
                    for token in tokens:
                        if token not in stopwords:
                            new_list.append(token)
                    return new_list
                
                df_sw['content'] = df_sw['content'].apply(stopword_removal)

                #stemming
                df_stem = df_sw.copy()
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                def stemming(text_cleaning):
                    do = []
                    for w in text_cleaning:
                        dt = stemmer.stem(w)
                        do.append(dt) 
                    d_clean = []
                    d_clean = " ".join(do)
                    return d_clean
                
                df_stem['content'] = df_stem['content'].apply(stemming)

                df_filtered = df_stem[df_stem['content'] != ''] 
                st.write("Dataframe setelah preprocessing:")
                st.dataframe(df_filtered)
                st.session_state.df_final = df_filtered
                return df_filtered
        else:
            st.error("Kolom 'content' tidak ditemukan dalam dataframe.")
            return df
    return None

        


def classification(df):
    with st.spinner("Sedang melakukan klasifikasi..."):
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        
        df['combined'] = df['content'].apply(lambda x: ''.join(x))
        x = df['combined'].sum()
        tfidf = vectorizer.transform(df['combined']).toarray()

        predictions = model.predict(tfidf)
        df_classified = df.copy()
        df_classified['prediction'] = predictions
        df_classified['prediction'] = df_classified['prediction'].apply(lambda x: 'Positif' if x == 1 else 'Negatif')

        df_classified.drop(['content', 'at'], axis='columns', inplace=True)

        st.session_state.klasifikasi = df_classified

        st.write("Dataframe hasil klasifikasi:")
        st.dataframe(df_classified)
        st.write("Visualisasi jumlah prediksi:")
        if 'prediction' in df_classified.columns:
            prediksi_counts = df_classified['prediction'].value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=prediksi_counts.index, y=prediksi_counts.values, palette='viridis')
            plt.title('Distribusi Hasil Klasifikasi')
            plt.xlabel('Prediksi')
            plt.ylabel('Jumlah')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

# Navigasi sidebar
with st.sidebar:
    selected = option_menu('Navigasi',
                ['Dashboards', 'Upload dan Tampilkan CSV', 'Preprocessing Data', 'Klasifikasi'], default_index=0)

# Halaman untuk mengupload dan menampilkan CSV
if selected == 'Dashboards':
    st.title('Dashboards')

    st.subheader('', divider='violet')
    st.subheader("Deskripsi Aplikasi:")
    st.write("Aplikasi ini dirancang untuk menganalisis sentimen ulasan pengguna terhadap aplikasi MyIM3 yang diunduh dari platform Google Play Store. Dengan memanfaatkan dataset yang tersedia di Kaggle, aplikasi ini menggunakan teknik pengumpulan data sekunder. Sentimen analisis ini dilakukan untuk memahami persepsi dan pengalaman pengguna terhadap aplikasi MyIM3.")
    
    st.subheader('', divider='violet')
    st.subheader("Dataset Overview:")
    st.write("Dataset yang digunakan berasal dari Kaggle.com dengan rentang waktu 10 Januari 2024 hingga 31 Januari 2024")
    df_data = pd.read_csv('datafix.csv', encoding='unicode_escape', sep=';')
    st.dataframe(df_data)

    st.subheader('', divider='violet')
    st.subheader("Data Train Overview:")
    st.write("Dalam membangun model machine learning, kami telah membagi dataset menjadi data train dan data test. Berikut adalah overview terhadap data yang digunakan dalam proses training model")
    df_train = pd.read_csv('data_train.csv', encoding='unicode_escape')
    st.dataframe(df_train)

    st.subheader('', divider='violet')
    st.subheader("Confusion Matrix")
    st.write("Pada tahap training machine learning kami, didapatkan confusion matrix sebagai berikut")
    df_confusion = pd.read_csv('confusion_matrix.csv', encoding='unicode_escape', index_col='P\R')
    df_hasil = pd.read_csv('hasil_model.csv', encoding='unicode_escape', index_col=0)
    st.dataframe(df_confusion)
    st.write("Dengan K-Fold Cross Validation 5 Fold, menghasilkan hasil terbaik sebagai berikut ")
    st.dataframe(df_hasil)

if selected == 'Upload dan Tampilkan CSV':
    st.title('Upload dan Tampilkan CSV')
    if st.session_state.df.empty:
        upload_and_display_csv()
    else:
        st.dataframe(st.session_state.df, width=3000, height=800)

if selected == 'Preprocessing Data':
    st.title('Preprocessing Data')
    if not st.session_state.df.empty:
        if st.button('Preprocess Data'):
            df = st.session_state.df
            df_processed = preprocess_data(df)
        elif not st.session_state.df_final.empty: # Melihatkan hasil preprocessing terakhir jika pindah tab
            st.dataframe(st.session_state.df_final)
    else:
        st.write("Mohon upload data terlebih dahulu")

if selected == 'Klasifikasi':
    st.title('Klasifikasi')
    if not st.session_state.df_final.empty:
        if st.button('Klasifikasi'):
            classification(st.session_state.df_final)
        elif not st.session_state.klasifikasi.empty:
            st.dataframe(st.session_state.klasifikasi)
            st.write("Visualisasi Persebaran Data")
    else:
        st.write("Mohon lakukan preprocessing terlebih dahulu")