# coding: utf-8
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from tensorflow.python.framework import ops
import threading
import ctypes

import train_service

app = Flask(__name__)
CORS(app)
threadsDic = {
    "key": "value"
}


@app.route("/predict", methods=['POST'])
def predictAPI():
    payload = json.loads(request.data)
    text = payload["text"]
    username = payload["username"]
    userId = payload["user_id"]
    intentIds = payload["intent_ids"]
    
    return jsonify(train_service.predict(text, username, userId, intentIds))

@app.route("/train", methods=['POST'])
def trainAPI():
    payload = json.loads(request.data)
    userId = payload["user_id"]
    trainingHistoryId = payload["training_history_id"]

    existThread = threadsDic.get(userId)
    if (existThread != None):
        terminate_thread(existThread)

    thread = threading.Thread(target=train_service.train, args=(payload,))
    thread.start()
    threadsDic[userId] = thread

    return jsonify({
        "trainingHistoryId": trainingHistoryId
    })

@app.route("/")
def index():
    return "Welcome to our chatbot!"

def terminate_thread(thread):
    """Terminates a python thread from another thread.
    :param thread: a threading.Thread instance
    """
    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

if __name__ == "__main__":
    app.run(threaded=True, host="0.0.0.0", port=5001)