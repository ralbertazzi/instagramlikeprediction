from keras.models import Sequential
from keras.layers import Dense
from mlxtend.data import autompg_data
import numpy as np

def autompg():
    X, y = autompg_data()
    X = X[:,:7]
    y /= 100.0
    for i in range(7):
        X[:,i] /= max(X[:,i])
    
    X_train, X_test = np.split(X, [300])
    y_train, y_test = np.split(y, [300])
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = autompg()

# Build the model
model = Sequential()
model.add(Dense(40, input_dim=7, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape'])

# Fit the model
model.fit(X_train, y_train, epochs=1500, batch_size=30, validation_data=(X_test, y_test))

# Evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
