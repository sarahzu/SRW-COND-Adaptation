from twarc import Twarc
import json

cred = json.load(open("./twitter_credentials"))
outfile = "./follower_ids"



t = Twarc(cred["api_key"], cred["cons_sec"], cred["acc_token"], cred["acc_token_sec"])

with open(outfile, 'w') as outfile:
    for follower in t.follower_ids("realDonaldTrump"):
        outfile.write(follower + "\n")
