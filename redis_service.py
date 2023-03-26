import redis

# redis
r = redis.Redis(
host='redis-10810.c263.us-east-1-2.ec2.cloud.redislabs.com',
port=10810,
password='DP52KjeuwAhZyjKrpvP3kBlOurft4mHi')

#Prefix
USER_ID_PREFIX_ = "userIdPrefix_"
SCRIPT_ID_PREFIX_ = "scriptIdPrefix_"
COLON = ":"

def set(key, value):
    r.set(key, value)

def get(key):
    return r.get(key)
