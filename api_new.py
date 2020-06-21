import tensorflow as tf
import keras
from keras.models import model_from_json
import emoji
import numpy as np
import pandas as pd


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }

from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import Sequential

model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64,input_shape=(10,50)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))

global graph
graph = tf.get_default_graph()

from keras.models import load_model
model = load_model('services/emojifier/model_new.h5')

##with open("services/emojifier/model.json", "r") as file:
##    model = model_from_json(file.read())
    
model.load_weights("services/emojifier/model.hdf5")
model._make_predict_function()

embeddings = {}
with open('services/emojifier/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs

def getOutputEmbeddings(X):
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]

    return embedding_matrix_output


def predict(string):
    X = pd.Series([string])
    print(X)
    emb_X = getOutputEmbeddings(X)
    print(emb_X.shape)
    with graph.as_default():
        session = keras.backend.get_session()
        init = tf.global_variables_initializer()
        session.run(init)
        keras.backend.set_session(session)

        print(model.predict(emb_X))
        p = model.predict_classes(emb_X)
        print(model.predict_classes(emb_X))
        return emoji.emojize(emoji_dictionary[str(p[0])])

if __name__ == "__main__":
    print(predict('i love to play baseball'))
