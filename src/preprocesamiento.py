from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

from nltk.util import ngrams
from collections import Counter


def detectar_outliers(textos):

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(textos)

    modelo = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    predicciones = modelo.fit_predict(X)

    return predicciones


def generar_ngramas(textos, n=2, top=10):

    todas = []

    for texto in textos:

        palabras = texto.split()

        grams = ngrams(palabras, n)

        todas.extend(grams)

    contador = Counter(todas)

    return contador.most_common(top)
