from twarc import Twarc
import json

cred = json.load(open("./twitter_credentials"))

outfile = "./follower_ids"
outfile2 = "./follower_info"


header = "id,name,sceen_name,location,followers_count,friends_count,favourites_count,created_at,language"


t = Twarc(cred["api_key"], cred["cons_sec"], cred["acc_token"], cred["acc_token_sec"])

def download_users(name_of_starting_node):
    with open(outfile, 'w') as out:
        for follower in t.follower_ids(name_of_starting_node):
            out.write(follower + "\n")


def get_attributes_on_users():
    with open(outfile, 'r') as inp, open(outfile2, 'w') as outp:
        outp.write(header + '\n')
        outp.flush()
        lines = inp.readlines()
        for user_info in t.user_lookup(lines):
            if user_info['location'] != '':
                temp_info = str(user_info['id'])+',' + user_info['name']+','\
                    + user_info['screen_name'] + ','+  user_info['location'] + ','\
                    + str(user_info['followers_count']) + ','+ str(user_info['friends_count'])+','\
                    + str(user_info['favourites_count']) + ',' + user_info['created_at'] + ',' + user_info['lang']
                outp.write(temp_info+'\n')
                outp.flush()

#download_users("realDonaldTrump")

get_attributes_on_users()
