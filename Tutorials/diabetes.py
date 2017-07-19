from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def diabetes():
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,:8]
    y = dataset[:,8]
    X_train, X_test = np.split(X, [600])
    y_train, y_test = np.split(y, [600])
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = diabetes()

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=1500, batch_size=30, validation_data=(X_test, y_test))

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
