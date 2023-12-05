import os
import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import re
from gensim.models import Word2Vec
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
from bs4 import BeautifulSoup


tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')

# Функции для очистки и преобразования csv в список слов

def review_to_wordlist(review, remove_stopwords=False ) -> str:
    # убираем ссылки вне тегов
    review = re.sub(r"http[s]?://(?:[f-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", review)
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^а-яА-Я]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = stopwords.words("russian")
        words = [w for w in words if not w in stops]
    return(words)

def review_to_sentences(review, tokenizer, remove_stopwords=False) -> list:
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

# path for semantic module dirrectory
current_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))

# path for models
models_path = os.path.join(current_dir, 'models')

# path for data
data_path = os.path.join(current_dir, 'training_data')

#path for file
file_path = os.path.join(data_path, 'comments_vk.txt')
print(file_path)

# converting data frm csv to text
with open(file_path, 'a', encoding='utf-8') as file:
  with open(os.path.join(data_path, 'posts_spb_today.csv'), 'r', encoding='utf-8') as data:
    for row in data:
      file.write(row)

# first stage text cleaning
with open(file_path, encoding='utf-8') as f:
    data = ''.join(i for i in f.read() if not i.isdigit())

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(data)

# second stage cleaning data
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub('\n', ' ', text)
sents = sent_tokenize(text)

punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~„“«»†*—/\-‘’'
clean_sents = []

for sent in sents:
    s = [w.lower().strip(punct) for w in sent.split()]
    clean_sents.append(s)

# checking first two sentences
print(clean_sents[:2])

# Initializing model (parametres could be revized)
model = Word2Vec(clean_sents, vector_size = 10, window=5,min_count = 1)

# checking data
data = pd.read_csv(os.path.join(data_path, "posts_spb_today.csv"), delimiter=";")

print(len(data))
print(data.head())

# getting list of sentences
sentences = []

print("Parsing sentences from training set...")
for review in data["text"]:
    sentences += review_to_sentences(review, tokenizer)

# checking list of sentences
print(len(sentences))
print(sentences[0])

# trainig model
print("Training model...")

custom_model = word2vec.Word2Vec(sentences, workers=4, vector_size=300, min_count=10, window=10, sample=1e-3)

# checking words in model
print(len(custom_model.wv.key_to_index))

# checking similarity of words
print(custom_model.wv.similarity('снег', 'тротуар'))

# path to save custom model
custom_model_path = os.path.join(models_path, 'default_trained.model')

# sacing custom model
print("Saving model...")
custom_model.save(custom_model_path)

#training customly pretrainde
model_trained = gensim.models.Word2Vec.load(custom_model_path)

model_trained.build_vocab(clean_sents, update=True)
model_trained.train(sentences, total_examples=model_trained.corpus_count, epochs=5)

# checking on result
print(model_trained.wv.similarity('снег', 'тротуар'))


