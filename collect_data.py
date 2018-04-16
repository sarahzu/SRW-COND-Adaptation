from twarc import Twarc
import json
import pandas as pd
import numpy as np

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

def get_id_from_username():

    #https://github.com/klout/opendata/blob/master/political_leaning/README.md
    with open("./data/political_leaning.csv") as input, open("./data/political_leaning_with_ids.csv",'w+') as output:
        users = pd.read_csv(input)
        users.set_index(users.columns[0])
        all_users = []
        for i, user in users.iterrows():
            all_users.append(user.user_name)

        temp = {}
        for user in t.user_lookup(all_users,id_type="screen_name"):
            temp[user['screen_name']] = user['id']



        twitter_dict = pd.DataFrame(list(temp.items()), columns=['user_name','id'])
        twitter_dict.set_index(twitter_dict.columns[0])
        new = pd.merge(users, twitter_dict, on='user_name', how='left')
        new = new[np.isfinite(new['id'])]
        new = new[['id','user_name','leaning']]
        new.to_csv(output,index=False)








def get_lists():
    t.list_members()

#download_users("realDonaldTrump")

#get_attributes_on_users()
get_id_from_username()
