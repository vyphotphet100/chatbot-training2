import json
import shutil
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import requests
stemmer = LancasterStemmer()

import numpy
import tflearn
from tensorflow.python.framework import ops
import pickle
import os;

# jServer = "http://localhost:8080"
jServer = "https://chatbot-vapt.herokuapp.com"

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predictAPI():
    payload = json.loads(request.data)
    text = payload["text"]
    username = payload["username"]
    intent_result = predict(text, username)

    return jsonify({
        "intentId": intent_result[0],
        "intentName": intent_result[1],
        "accuracy": str(intent_result[2])
    })

@app.route("/train", methods=['POST'])
def trainAPI():
    payload = json.loads(request.data)
    userId = payload["user_id"]
    username = payload["username"]
    trainingHistoryId = payload["training_history_id"]

    # Tao folder tmp 
    if os.path.exists(username) and not os.path.exists(username + "-tmp"):
        shutil.copytree(username, username + "-tmp")

    if not os.path.exists(username):
        os.makedirs(username)

    with open(username + "/intents.json", "w+") as outfile:
        outfile.write(json.dumps(payload, indent=4))

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in payload["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern["content"])
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["name"])

        if intent["name"] not in labels:
            labels.append(intent["name"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

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

    with open(username + "/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(username + "/model.tflearn")

    # Gui tin hieu train xong
    requestBody = json.dumps({
        'user_id': userId,
        'training_history_id': trainingHistoryId
    })

    headers = {
        'Content-Type': 'application/json'
    }

    requests.request("POST", jServer + "/api/training/train_done", data=requestBody, headers=headers)

    if os.path.exists(username + "-tmp"):
        shutil.rmtree(username + "-tmp")

    return jsonify({
        "trainingHistoryId": trainingHistoryId
    })

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def predict(text, username):
    folderPath = username
    if (os.path.exists(username + "-tmp")):
        folderPath += "-tmp"

    # Load model
    with open(folderPath + "/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    ops.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.load(folderPath + "/model.tflearn")

    # Load words và labels
    words = []
    labels = []

    with open(folderPath + "/intents.json") as file:
        data = json.load(file)

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern["content"])
            words.extend(wrds)

        if intent["name"] not in labels:
            labels.append(intent["name"] + "|" + intent["id"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # predict
    results = model.predict([bag_of_words(text, words)])
    results_index = numpy.argmax(results)
    tagWithId = labels[results_index]
    tag = tagWithId.split("|")[0]
    intentId = tagWithId.split("|")[1]
    return [intentId, tag, results[0][results_index]]


@app.route("/")
def index():
    return "Welcome to our chatbot!"

if __name__ == "__main__":
    app.run(threaded=True)