import redis

# redis
r = redis.Redis(
host='localhost',
port=6380)

#Prefix
USER_ID_PREFIX_ = "userIdPrefix_"
SCRIPT_ID_PREFIX_ = "scriptIdPrefix_"
COLON = ":"

def set(key, value):
    r.set(key, value)

def get(key):
    return r.get(key)
