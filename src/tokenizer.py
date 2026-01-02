import jieba
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

jieba.initialize()

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize(text):
    text = str(text).lower()

    # 去 URL
    text = re.sub(r"http\S+|www\S+", "", text)

    # 只保留：中文 + 英文 + 数字
    text = re.sub("[^a-zA-Z0-9\u4e00-\u9fa5]", " ", text)

    tokens = []

    # 1️⃣ 中文分词（jieba）
    chinese_part = re.findall("[\u4e00-\u9fa5]+", text)
    for chunk in chinese_part:
        tokens.extend(jieba.lcut(chunk))

    # 2️⃣ 英文分词 + 词形还原
    english_part = re.findall("[a-zA-Z]+", text)
    if english_part:
        pos_tags = nltk.pos_tag(english_part)
        for word, tag in pos_tags:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            tokens.append(lemma)

    return tokens
