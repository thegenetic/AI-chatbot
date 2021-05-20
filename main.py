from tensorflow.python.framework import ops
from pandas.core.dtypes.common import classes
import json
import pickle
import random
import numpy
import tensorflow as tf
import os
import nltk
import tflearn
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

try:  # if want to change the model then put an x underneath
    # pass
    with open("data.pickel", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)  # needed to train the model
            docs_y.append(intent["tag"])  # needed to train the model

        if intent['tag'] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("modle.tflearn")
# if os.path.exists("model.tflearn.meta"):
#     model.load("model.tflearn")
# else:
#     model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#     model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower())for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot(type quite to stop)")
    while True:
        inp = input("You :")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        # print(result)
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print(tag)


chat()