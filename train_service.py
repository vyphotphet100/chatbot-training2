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


def train(payload):
    userId = payload["user_id"]
    username = payload["username"]
    scriptId = payload["script_id"]
    modelPath = username + "/" + scriptId

    # Tạo folder lưu model nếu chưa có 
    if not os.path.exists(username):
        os.makedirs(username)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    with open(modelPath + "/intents.json", "w+") as outfile:
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
    payload = None

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

    with open(modelPath + "/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save(modelPath + "/model.tflearn")

    # Gui tin hieu train xong
    redis_service.set(userId + ":training_server_status", "free")
    app.threadsDic[userId] = None

    # set hết model lên redis
    redisKey = redis_service.USER_ID_PREFIX_ + userId + redis_service.COLON + redis_service.SCRIPT_ID_PREFIX_ + scriptId
    with open(modelPath + "/checkpoint", "rb") as file:
        base64_checkpoint = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:checkpoint", base64_checkpoint)

    with open(modelPath + "/data.pickle", "rb") as file:
        base64_data_pickle = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:data.pickle", base64_data_pickle)

    with open(modelPath + "/intents.json", "rb") as file:
        base64_intents_json = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:intents.json", base64_intents_json)

    with open(modelPath + "/model.tflearn.data-00000-of-00001", "rb") as file:
        base64_model_tflearn_data_00000_of_00001 = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:model.tflearn.data-00000-of-00001", base64_model_tflearn_data_00000_of_00001)

    with open(modelPath + "/model.tflearn.index", "rb") as file:
        base64_model_tflearn_index = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:model.tflearn.index", base64_model_tflearn_index)

    with open(modelPath + "/model.tflearn.meta", "rb") as file:
        base64_model_tflearn_meta = base64.b64encode(file.read())
        redis_service.set(redisKey + ":model_file:model.tflearn.meta", base64_model_tflearn_meta)

    # Tăng version cho model 
    modelVersion = redis_service.get(redisKey + ":model_file:version")
    if (modelVersion == None or modelVersion == ""):
        open(modelPath + "/version", "wt").write('1')
        redis_service.set(redisKey + ":model_file:version", str("1"))
    else:
        modelVersion = int(modelVersion)
        redis_service.set(redisKey + ":model_file:version", str(modelVersion + 1))
        open(modelPath + "/version", "wt").write(str(modelVersion + 1))

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

def predict(text, username, userId, intentIds, scriptId):
    folderPath = username + "/" + scriptId + "-predict"
    redisKey = redis_service.USER_ID_PREFIX_ + userId + redis_service.COLON + redis_service.SCRIPT_ID_PREFIX_ + scriptId

    # Kiểm tra xem có training thread của user này có đang chạy hay không 
    existThread = app.threadsDic.get(userId)
    if (existThread == None):
        redis_service.set(userId + ":training_server_status", "free")

    # load model mới từ redis về  nếu nó đã được train mới (version tăng)
    modelVersion = redis_service.get(redisKey + ":model_file:version")
    if (modelVersion == None or modelVersion == ""): # model này chưa được train 
        return None

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        saveModelFromRedis(redisKey, folderPath)
    else:
        modelVersionInFile = None
        with open(folderPath + "/version", "rt") as file:
            modelVersionInFile = file.readline()
        if (int(modelVersionInFile) != int(modelVersion)):
            saveModelFromRedis(redisKey, folderPath)
    
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

    data = None
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
    maxAccuracy = 0
    resultIdx = -1
    for result in results:
        for i in range(0, len(result)):
            tagWithId = labels[i]
            tag = tagWithId.split("|")[0]
            intentId = tagWithId.split("|")[1]
            
            if (intentId not in intentIds):
                continue

            if (result[i] > maxAccuracy and result[i] > 0.5):
                maxAccuracy = result[i]
                resultIdx = i


    if (resultIdx == -1):
        return [None, None, 0]

    tagWithId = labels[resultIdx]
    tag = tagWithId.split("|")[0]
    intentId = tagWithId.split("|")[1]
    return [intentId, tag, results[0][results_index]]

def saveModelFromRedis(redisKey, folderPath):
    base64_checkpoint = redis_service.get(redisKey + ":model_file:checkpoint")
    base64_data_pickle = redis_service.get(redisKey + ":model_file:data.pickle")
    base64_intents_json = redis_service.get(redisKey + ":model_file:intents.json")
    base64_model_tflearn_data_00000_of_00001 = redis_service.get(redisKey + ":model_file:model.tflearn.data-00000-of-00001")
    base64_model_tflearn_index = redis_service.get(redisKey + ":model_file:model.tflearn.index")
    base64_model_tflearn_meta = redis_service.get(redisKey + ":model_file:model.tflearn.meta")
    modelVersion = int(redis_service.get(redisKey + ":model_file:version"))

    # save file 
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    with open(folderPath + "/checkpoint", "wb") as fh:
        fh.write(base64.decodebytes(base64_checkpoint))
    with open(folderPath + "/data.pickle", "wb") as fh:
        fh.write(base64.decodebytes(base64_data_pickle))
    with open(folderPath + "/intents.json", "wb") as fh:
        fh.write(base64.decodebytes(base64_intents_json))
    with open(folderPath + "/model.tflearn.data-00000-of-00001", "wb") as fh:
        fh.write(base64.decodebytes(base64_model_tflearn_data_00000_of_00001))
    with open(folderPath + "/model.tflearn.index", "wb") as fh:
        fh.write(base64.decodebytes(base64_model_tflearn_index))
    with open(folderPath + "/model.tflearn.meta", "wb") as fh:
        fh.write(base64.decodebytes(base64_model_tflearn_meta))
    open(folderPath + "/version", "wt").write(str(modelVersion))


