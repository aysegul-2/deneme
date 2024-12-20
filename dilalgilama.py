import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Load your data (replace "dataset.csv" with your actual file path)
data = pd.read_csv("dataset.csv")

X = np.array(data["Text"])
y = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

def main():
  st.title("Dil Tahmin Uygulaması")

  # Kullanıcıdan metin girişi al
  text_input = st.text_area("Lütfen tahmin etmek istediğiniz metni girin:")

  # Girilen metni vektörleştir
  X_new = cv.transform([text_input])

  # Model ile tahmin yap
  prediction = model.predict(X_new)[0]

  # Sonucu göster
  st.write("Tahmin edilen dil:", prediction)

if __name__ == "__main__":
  main()