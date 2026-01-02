from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(texts, max_df, min_df, max_features):
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer
