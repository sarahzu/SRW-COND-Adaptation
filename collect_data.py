from twarc import Twarc
import json
import pandas as pd
import numpy as np
import networkx as nx

cred = json.load(open("./twitter_credentials"))
t = Twarc(cred["api_key"], cred["cons_sec"], cred["acc_token"], cred["acc_token_sec"])


def download_users(name_of_starting_node):
    with open("./data/follower_ids.csv", 'w+') as out:
        for follower in t.follower_ids(name_of_starting_node):
            out.write(follower + "\n")


def get_attributes_on_users():
    with open("./data/follower_ids.csv", 'r') as inp, open("./data/follower_info.csv", 'w+') as outp:
        outp.write("id,name,sceen_name,location,followers_count,friends_count,created_at,language" + '\n')
        outp.flush()
        lines = inp.readlines()
        for user_info in t.user_lookup(lines):
            if user_info['location'] != '':
                temp_info = str(user_info['id'])+',' + user_info['name']+','\
                    + user_info['screen_name'] + ','+  user_info['location'] + ','\
                    + str(user_info['followers_count']) + ','+ str(user_info['friends_count'])+','\
                    + ',' + user_info['created_at'] + ',' + user_info['lang']
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


def get_followers():
    with open("./data/political_leaning_with_ids.csv",'r') as input, open("./data/followers.csv",'w+') as output:
        ids = [int(elem) for elem in list(pd.read_csv(input)['id'])]
        users_followers = pd.DataFrame(columns=['followers','friends'],index=ids)
        for user in ids[2:4]:
            followers = []
            friends = []
            for id in t.follower_ids(user):
                if id in ids:
                    followers.append(id)
            for id in t.friend_ids(user):
                if id in ids:
                    friends.append(id)
            users_followers.at[user,'followers'] = followers
            users_followers.at[user,'friends'] = friends
        users_followers.to_csv(output)



def create_network_graph():
    def add_node(graph, node, label):
        node = int(node)
        label = str(label)
        if not graph.has_node(node):
            graph.add_node(node, attr_dict={'weight': 1, 'label': label})
        else:
            graph.node[node]['weight'] += 1
            graph.node[node]['label'] = label

    def add_edge(graph, node_1, label_1, node_2, label_2):
        node_1 = int(node_1)
        node_2 = int(node_2)
        label_1 = str(label_1)
        label_2 = str(label_2)

        if not graph.has_edge(node_1, node_2):
            if not graph.has_node(node_1):
                graph.add_node(node_1, attr_dict={'weight': 1, 'label': label_1})
            if not graph.has_node(node_2):
                graph.add_node(node_2, attr_dict={'weight': 1, 'label': label_2})
            graph.add_edge(node_1, node_2, attr_dict={'weight': 1})
        else:
            raph[node_1][node_2]['weight'] += 1

    def write_graphml(graph, output_path):
        print("Writing Graph to " + str(output_path))
        nx.write_graphml(graph, output_path)

    def print_info(graph):
        print('--------------------------------------')
        print('Number of nodes: ' + str(graph.number_of_nodes()))
        print('Number of edges: ' + str(graph.number_of_edges()))
        print('Average degree: ' + str(
            round(float(sum(graph.degree().values())) / float(graph.number_of_nodes()), 2)))

        avg_sum = 0
        for _, data in graph.nodes(data=True):
            avg_sum += data['weight']

        print('Average weight: ' + str(round(float(avg_sum) / float(graph.number_of_nodes()), 2)))
        print('--------------------------------------')

    with open("./data/followers.csv",'r') as input:
        dataframe = pd.read_csv(input)
        dataframe = dataframe[dataframe['followers'] != []]

    graph = nx.Graph()
    for _,row in dataframe.iterrows():
        add_node(graph,row['id'],str(row['id']))
        for follower in row['followers']:
            add_edge(graph,row['id'],str(row['id']),follower,str(follower))


    print_info(graph)
    write_graphml(graph,"test_graph.graphml")


get_followers()


#download_users("realDonaldTrump")

#get_attributes_on_users()
#get_id_from_username()
