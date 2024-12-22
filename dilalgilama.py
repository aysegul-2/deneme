import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# CSV dosyasının local path'i
CSV_PATH  = pd.read_csv("https://raw.githubusercontent.com/aysegul-2/deneme/main/dataset.csv")


# 1. Model eğitimi (önceden CSV ile)
def train_model():
    data = pd.read_csv(CSV_PATH)
    
    if "Text" not in data.columns or "language" not in data.columns:
        raise ValueError("CSV dosyasında 'Text' ve 'language' sütunları olmalıdır.")
    
    cv = CountVectorizer()
    X = cv.fit_transform(data["Text"])
    y = data["language"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    with open("language_model.pkl", "wb") as f:
        pickle.dump((model, cv), f)
    
    return model, cv

# 2. Model yükleme
@st.cache_resource
def load_model():
    try:
        with open("language_model.pkl", "rb") as f:
            model, cv = pickle.load(f)
    except FileNotFoundError:
        model, cv = train_model()
    return model, cv

# 3. Ana uygulama
def main():
    st.title("Dil Tahmin Uygulaması")
    
    # Modeli yükle
    model, cv = load_model()

    # Kullanıcıdan metin girişi al
    text_input = st.text_area("Lütfen tahmin etmek istediğiniz metni girin:")

    if st.button("Tahmin Et"):
        if text_input.strip():  # Metin boş değilse işlem yap
            # Metni vektörleştir
            X_new = cv.transform([text_input])
            
            # Model ile tahmin yap
            prediction = model.predict(X_new)[0]
            
            # Sonucu göster
            st.write("Tahmin edilen dil:", prediction)
        else:
            st.warning("Lütfen tahmin etmek için bir metin girin!")

if __name__ == "__main__":
    main()

