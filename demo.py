import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Dense, SimpleRNN, Flatten, Embedding

samples = ['The cat sat on the mat.',
           'The dog ate my homework.']

def one_hot_by_word():
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples),
                              max_length,
                              max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1

    return results

def one_hot_by_keras():
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(samples)

    sequences = tokenizer.texts_to_sequences(samples)

    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
    print(len(one_hot_results))

def rnn():
    max_features = 10000
    maxlen = 500
    batch_size = 32
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    print(x_train)

def imdb_detail():
    (x0, y0), (x1, y1) = imdb.load_data(num_words=50)
    index = imdb.get_word_index()
    word_index = dict([(v, k) for (k, v) in index.items()])
    code = ' '.join([word_index.get(i-3, '?') for i in x0[0]])
    print(index)
    print(code)

def gpu_version():
    pass

def demo_model():
    model = Sequential()


if __name__ == '__main__':
    rnn()


