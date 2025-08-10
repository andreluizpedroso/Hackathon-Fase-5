from __future__ import annotations
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Stopwords PT-BR básicas (lista pequena e estável para evitar downloads)
PORTUGUESE_STOPWORDS: List[str] = [
    "a","o","os","as","de","da","do","das","dos","e","é","em","para","por","com","sem",
    "um","uma","uns","umas","no","na","nos","nas","ao","à","aos","às","se","que","quem",
    "qual","quais","quando","onde","como","porque","porquê","mas","ou","também","muito",
    "muita","muitos","muitas","pouco","pouca","poucos","poucas","ser","ter","estar","vai",
    "vou","foi","era","são","sua","seu","suas","seus","ele","ela","eles","elas","você",
    "vocês","nosso","nossa","nossos","nossas","meu","minha","meus","minhas","deve","devem",
    "há","haver","entre","sobre","até","após","antes","depois"
]

def build_pipeline(max_features: int = 40000) -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        stop_words=PORTUGUESE_STOPWORDS,  # <- aqui
        max_features=max_features
    )
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])
    return pipe
