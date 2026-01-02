def load_stopwords(*paths):
    stopwords = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            stopwords |= set(line.strip() for line in f)
    return stopwords

def remove_stopwords(tokens, stopwords):
    return [w for w in tokens if w not in stopwords and len(w) > 1]
