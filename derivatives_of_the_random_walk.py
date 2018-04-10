from twarc import Twarc
import json

cred = json.load(open("./twitter_credentials"))


t = Twarc(cred["api_key"], cred["cons_sec"], cred["acc_token"], cred["acc_token_sec"])
for tweet in t.search("ferguson"):
    print(tweet)
