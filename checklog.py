# import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.preprocessing import sequence

gram = {}

def ngram(chars, gram_len=2):
    l = len(chars)
    c = []
    for i in range(l-gram_len+1):
        substr = chars[i:i+gram_len]
        if substr not in gram:
            gram[substr] = len(gram) + 1
        c.append(gram[substr])
    return c

def read_logs(path):
    files = os.listdir(path)
    css = []
    for file in files:
        if file.find('.log') > -1:
            target_path = '{}/{}'.format(path, file)
            with open(target_path, 'r') as f:
                content = f.readlines()
                cs = [ngram(c) for c in content]
                css = css + cs
                f.close()
    return css

def main():
    # mock_data.generate_simple_time_text_log()
    cs = read_logs('./logs/whites')
    N = max([len(c) for c in cs])
    xn = sequence.pad_sequences(cs, maxlen=N)
    vn = np.ones(len(cs),)

    cs_tests = read_logs('./logs/tests')
    xn_tests = sequence.pad_sequences(cs_tests, maxlen=N)
    vn_tests = np.random.randint(0,2,len(cs))

    print(xn.shape)
    print(vn.shape)

    model = Sequential()
    model.add(Dense(64, input_shape=(N,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    # model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.fit(xn, vn, epochs=4, batch_size=64)

    results = model.predict(xn_tests)
    print(results)


if __name__ == '__main__':
    main()