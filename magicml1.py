# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# %%
cols =["flength", "fwidth", "fsize","fcone","fcone1","fasym","fm3long","fm3trans","faplha","fdist","class"]
df= pd.read_csv("magic04.data", names= cols)
df.head()

# %%
#g= gamma and h = hadron

df["class"]= (df["class"]== "g").astype(int)

# %%
 df.head()

# %%
for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

# %%
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# %%
train = pd.DataFrame(train)

# %%
len(train)

# %%
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y

# %%
train

# %%
len(train)

# %%
print(len(train[train["class"]==1]))
print(len(train[train["class"]==0]))

# %%
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

# %%
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
knn_model = KNeighborsClassifier(n_neighbors= 5)
knn_model.fit(X_train, y_train)

# %%
y_pred = knn_model.predict(X_test)

# %%
y_test

# %%
print(classification_report(y_test,y_pred))

# %%
#naive bayes

from sklearn.naive_bayes import GaussianNB

# %%
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

# %%
y_pred = nb_model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
#logistic regression

from sklearn.linear_model import LogisticRegression

# %%
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

# %%
y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
#SVM 

from sklearn.svm import SVC

# %%
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

# %%
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
#neural network

import tensorflow as tf

# %%
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_accuracy(history):

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)

    plt.show()

# %%
history = nn_model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0
  )

# %%
plot_loss(history)
plot_accuracy(history)

# %%
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history

# %%
def plot_history(history):
    fig,(ax1, ax2) = plt.subplots(1,2)
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show(), 

# %%
least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [16, 32, 64]:
  for dropout_prob in[0, 0.2]:
    for lr in [0.01, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss = model.evaluate(X_valid, y_valid)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model

# %%
y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

# %%
print(classification_report(y_test, y_pred))

# %%



