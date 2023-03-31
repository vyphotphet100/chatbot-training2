import tflearn
from tensorflow.python.framework import ops
import os
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import json
import numpy
import pickle
import redis_service
import base64
import app
import shutil
import pandas as pd
import spacy
from wordcloud import WordCloud, STOPWORDS
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt
import re
import random
from spacy.training.example import Example


def train(payload):
    userId = payload["user_id"]
    username = payload["username"]
    modelPath = username

    # Tạo folder lưu model nếu chưa có 
    if not os.path.exists(username):
        os.makedirs(username)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    # Lưu payload vào file 
    with open(modelPath + "/intents.json", "w+") as outfile:
        outfile.write(json.dumps(payload, indent=4))

    words = [] # Chữ đã được tách của toàn bộ pattern
    intentCodes = [] # Code của các intent (unique)
    intentIds = [] # Id của các intent (unique)
    docsX = [] # [ ["tôi", "tên", "là", "vỹ"] , ["tôi", "tên", "là", "linh"] , ...]
    docsY = [] # Code intent của toàn bộ pattern (no-unique)
    for intent in payload["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern["content"])
            wrds = [stemmer.stem(w.lower()) for w in wrds]
            words.extend([w for w in wrds if (w not in words)])

            if ("entities" in pattern):
                for entity in pattern["entities"]:
                    wrds.append(entity["entity_type_id"])
                    if (entity["entity_type_id"] not in words):
                        words.append(entity["entity_type_id"])

            docsX.append(wrds)
            docsY.append(intent["code"])

        if intent["code"] not in intentCodes:
            intentCodes.append(intent["code"])
        if intent["id"] not in intentIds:
            intentIds.append(intent["id"])

    # Convert doc_x và doc_y về training và output (Dạng 0,1,1,0,...)
    training = []
    output = []
    out_empty = [0 for _ in range(len(intentCodes))]
    for x, wrds in enumerate(docsX):
        # Tạo training
        bag = []
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        training.append(bag)

        # Tạo output 
        output_row = out_empty[:]
        output_row[intentCodes.index(docsY[x])] = 1
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open(modelPath + "/intent_data.pickle", "wb") as f:
        pickle.dump((words, intentIds, intentCodes, training, output), f)

    ops.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(modelPath + "/intent_model.tflearn")

    # Train cho entity
    trainForEntityType(payload)

    # Gui tin hieu train xong
    redis_service.set(userId + ":training_server_status", "free")
    app.threadsDic[userId] = None
    print("TRAIN SUCCESSFULLY FOR USER: " + username)

    if (os.path.exists(modelPath + "-predict")):
        shutil.rmtree(modelPath + "-predict")
    os.rename(modelPath, modelPath + "-predict")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def trainForEntityType(payload):
    username = payload["username"]
    modelPath = username
    # NER
    spacy.load('en_core_web_sm')
    TRAIN_DATA = []
    for intent in payload["intents"]:
        for pattern in intent["patterns"]:
            ent_dict = {}
            content = pattern["content"].lower()
            entities = []
            if ("entities" not in pattern):
                continue

            for entity in pattern["entities"]:
                entityToTrain = (int(entity["start_position"]), int(entity["end_position"]) + 1, entity["entity_type"]["id"])
                entities.append(entityToTrain)
            if len(entities) > 0:
                ent_dict['entities'] = entities
                train_item = (content, ent_dict)
                TRAIN_DATA.append(train_item)
    
    # Let training
    nlp = train_ner(TRAIN_DATA)

    # Tạo folder lưu model nếu chưa có 
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    with open(modelPath + "/entity_nlp.pickle", "wb") as f:
        pickle.dump((nlp), f)

"""### Training the NER Model"""
def train_ner(training_data):
    """Steps
    Create a Blank NLP  model object
    Create and add NER to the NLP model
    Add Labels from your training data
    Train  
    """
    n_iter = 50
    TRAIN_DATA = training_data
    nlp = spacy.blank("en")  # create blank Language class
    
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)
    ner = nlp.get_pipe("ner")
        
    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
            
    nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        # batches = minibatch(TRAIN_DATA, size=compounding(1, len(TRAIN_DATA), 1.001))
        batches = minibatch(TRAIN_DATA, size=8)
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = []
            for i in range(0, len(texts)):
                doc = nlp.make_doc(texts[i])
                example = Example.from_dict(doc, annotations[i])
                examples.append(example)
            # Update the model
            nlp.update(examples, losses=losses, drop=0.3)
        print("Losses", losses)
        if (losses["ner"] < 3):
            break
    return nlp

def predict(text, username, userId, acceptIntentIds):
    folderPath = username + "-predict"

    # Kiểm tra xem có training thread của user này có đang chạy hay không 
    existThread = app.threadsDic.get(userId)
    if (existThread == None):
        redis_service.set(userId + ":training_server_status", "free")
    
    if not os.path.exists(folderPath):
        return None # model này chưa được train 

    # Load model
    with open(folderPath + "/intent_data.pickle", "rb") as f:
        words, intentIds, intentCodes, training, output = pickle.load(f)

    entities = predictEntityType(text, username, userId)
    if (entities != []):
        for entity in entities:
            text = text + " " + entity["entityTypeId"]

    ops.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.load(folderPath + "/intent_model.tflearn")

    # predict intent 
    results = model.predict([bag_of_words(text, words)])
    results_index = numpy.argmax(results)
    maxAccuracy = 0
    resultIdx = -1
    for result in results:
        for i in range(0, len(result)):
            if ((result[i] > maxAccuracy) and (result[i] > 0.3) and (intentIds[i] in acceptIntentIds)):
                maxAccuracy = result[i]
                resultIdx = i

    if (resultIdx == -1):
        return {
            "intentId": "-1",
            "intentCode": "-1",
            "acurracy": "-1",
            "entities": entities
        }

    intentId = intentIds[resultIdx]
    intentCode = intentCodes[resultIdx]
    return {
        "intentId": intentId,
        "intentCode": intentCode,
        "acurracy": str(results[0][resultIdx]),
        "entities": entities
    }

def predictEntityType(text, username, userId):
    folderPath = username + "-predict"

    if not os.path.exists(folderPath):
        return []

    nlp = None
    with open(folderPath + "/entity_nlp.pickle", "rb") as f:
        nlp = pickle.load(f)

    resEntities = []
    text.lower()
    doc = nlp(text)
    results = [(ent,ent.label_) for ent in doc.ents]
    for result in results:
        resEntities.append({
            "value": str(result[0]),
            "entityTypeId": str(result[1]),
            "startPosition": text.index(str(result[0])),
            "endPosition": text.index(str(result[0])) + len(str(result[0])) - 1
        })
    return resEntities

