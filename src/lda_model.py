from gensim import corpora
from gensim.models import LdaModel
from os import system

def train_lda(texts, num_topics, passes, chunksize=2000, update_every=1, iterations=50, eval_every=None):
    """
    Train an LDA model using gensim's LdaModel in online mode.

    Parameters:
    - texts: list of token lists
    - num_topics: number of topics
    - passes: number of passes through the corpus (each pass processes the corpus in chunks)
    - chunksize: number of documents to be used in each training chunk
    - update_every: how many chunks to process before updating the model (set 1 for online updates)
    - iterations: number of iterations per chunk
    - eval_every: how often to evaluate perplexity (None to disable)

    Returns:
    - trained LdaModel
    """
    dictionary = corpora.Dictionary(texts)
    print("Initial dictionary size:", len(dictionary)) # 90137
    dictionary.filter_extremes(no_below=15, no_above=0.6)
    print("Dictionary size after filtering:", len(dictionary)) # 12046
    # system("pause")
    corpus = [dictionary.doc2bow(t) for t in texts]

    # Use LdaModel in online mode by setting update_every=1 (update model every chunk)
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        chunksize=chunksize,
        update_every=update_every,
        passes=passes,
        iterations=iterations,
        eval_every=eval_every,
        alpha="auto",
        eta=0.01,
        random_state=42,
    )

    return lda, dictionary
