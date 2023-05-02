import pandas as pd
import string
from nltk.corpus import stopwords
import nltk

data = pd.read_csv("test.csv")
for col in ['DESCRIPTION', 'TITLE', 'BULLET_POINTS']:
    data[col] = data[col].astype(str).apply(lambda x: x.lower())

print('reached')
for col in ['DESCRIPTION', 'TITLE', 'BULLET_POINTS']:
    data[col] = data[col].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
  
print('reached')
stop = stopwords.words('english')
for col in ['DESCRIPTION', 'TITLE', 'BULLET_POINTS']:
    data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

data['TEXT'] = data['TITLE'] + ' ' + data['DESCRIPTION'] + ' ' + data['BULLET_POINTS']
data.drop('TITLE', axis=1, inplace=True)
data.drop('DESCRIPTION', axis=1, inplace=True)
data.drop('BULLET_POINTS', axis=1, inplace=True)

import gensim
review_text = data.TEXT.astype(str).apply(gensim.utils.simple_preprocess)
print("review text processed")
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)
import numpy as np
from tqdm import tqdm

w2v_model = gensim.models.Word2Vec.load('word2vec.model')
def text_to_vector(text):
    vector = np.zeros(w2v_model.vector_size)
    num_words = 0
    for word in text:
        if word in w2v_model.wv:
            vector += w2v_model.wv[word]
            num_words += 1
    if num_words > 0:
        vector /= num_words
    return vector

print("model loaded")
data['vectors'] = review_text.apply(lambda x: text_to_vector(x))
data.to_csv('/content/drive/MyDrive/MLSS/Amazon ML Challenge/preprocessedTest')