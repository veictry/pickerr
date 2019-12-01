import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

def perceptron(xn, yn, MaxIter=1000, a=0.1, w=np.random.rand(3,)):
    N = xn.shape[0]
    f = lambda x: np.sign(w[0]*1 + w[1]*x[0] + w[2]*x[1])

    for _ in range(MaxIter):
        i = np.random.randint(N)
        print(i)
        if yn[i] != f(xn[i,:]):
            w[0] = w[0] + yn[i] * 1 * a
            w[1] = w[1] + yn[i] * xn[i, 0] * a
            w[2] = w[2] + yn[i] * xn[i, 1] * a
    return w

def norm(l):
    pass

def build_model(xn, yn):
    x_train = xn
    y_train =(yn + 1) / 2.0
    model = Sequential()
    model.add(Dense(32, input_shape=(2,), activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(xn, yn, epochs=5, batch_size=10,validation_split=(x_train, y_train))


def main():
    fig = plt.figure()
    ax0 = plt.gca()

    N = 100
    pn = np.random.rand(N, 2)
    vn = np.zeros((N, 1))

    a = np.random.rand()
    b = np.random.rand()
    f = lambda x: a * x + b
    x = np.linspace(0, 1)
    plt.plot(x, f(x), 'r')

    for i in range(N):
        if f(pn[i,0]) >= pn[i,1]:
            vn[i] = 1
            plt.plot(pn[i,0], pn[i,1], 'bo', markersize=12)
        else:
            vn[i] = -1
            plt.plot(pn[i,0], pn[i,1], 'ro', markersize=12)

    w = perceptron(pn, vn)
    build_model()

    a = -w[0] / w[2]
    b = -w[1] / w[2]
    y = lambda x: a * x + b
    plt.plot(x, y(x), 'g--')

    plt.show()

if __name__ == '__main__':
    main()