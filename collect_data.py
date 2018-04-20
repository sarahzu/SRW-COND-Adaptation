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


def get_id_from_username():
    # https://github.com/klout/opendata/blob/master/political_leaning/README.md
    with open("./data/political_leaning.csv") as input, open("./data/political_leaning_with_ids.csv", 'w+') as output:
        users = pd.read_csv(input)
        users.set_index(users.columns[0])
        all_users = []
        for i, user in users.iterrows():
            all_users.append(user.user_name)

        temp = {}
        for user in t.user_lookup(all_users, id_type="screen_name"):
            temp[user['screen_name']] = int(user['id'])

        twitter_dict = pd.DataFrame(list(temp.items()), columns=['user_name', 'id'])
        twitter_dict.set_index(twitter_dict.columns[0])
        new = pd.merge(users, twitter_dict, on='user_name', how='left')
        new = new[np.isfinite(new['id'])]
        new = new[['id', 'user_name', 'leaning']]
        new.to_csv(output, index=False)


def get_attributes_on_users():
    with open("./data/political_leaning_with_ids.csv", 'r') as inp, open("./data/follower_info.csv", 'w+') as outp:
        outp.write("id,followers_count,friends_count,created_at,language" + '\n')
        outp.flush()
        dataframe = pd.read_csv(inp)
        ids = [str(int(elem)) for elem in dataframe['id'].tolist()]
        for user_info in t.user_lookup(ids):
            temp_info = str(user_info['id']) + ',' + str(user_info['followers_count']) + \
                        ',' + str(user_info['friends_count']) + ',' + user_info['created_at'] + ',' + user_info['lang']
            outp.write(temp_info + '\n')
            outp.flush()


def clean_data(threshold_followers_upper, threshold_followers_lower, threshold_friends_upper, threshold_friends_lower):
    with open("./data/follower_info.csv", 'r') as inp:
        dataframe = pd.read_csv(inp)

    dataframe.set_index('id')
    dataframe = dataframe[dataframe['followers_count'] < threshold_followers_upper]
    dataframe = dataframe[dataframe['followers_count'] > threshold_followers_lower]
    dataframe = dataframe[dataframe['friends_count'] < threshold_friends_upper]
    dataframe = dataframe[dataframe['friends_count'] > threshold_friends_lower]

    with open("./data/follower_info_cleaned.csv", 'w+') as outp:
        dataframe.to_csv(outp)


def get_follower_ids(id):
    user = str(id)
    url = 'https://api.twitter.com/1.1/followers/ids.json'
    params = {'user_id': user, 'cursor': -1, 'count': 100}
    try:
        resp = t.get(url, params=params, allow_404=True)
    except t.exceptions.HTTPError as e:
        raise e
    user_ids = resp.json()
    for user_id in user_ids['ids']:
        yield str(user_id)


def get_friends_ids(id):
    user = str(id)
    url = 'https://api.twitter.com/1.1/friends/ids.json'
    params = {'user_id': user, 'cursor': -1, 'count': 100}
    try:
        resp = t.get(url, params=params, allow_404=True)
    except t.exceptions.HTTPError as e:
        raise e
    user_ids = resp.json()
    for user_id in user_ids['ids']:
        yield str(user_id)


def get_followers():
    with open("./data/follower_info_cleaned.csv", 'r') as input, open("./data/network_data.csv", 'w+') as output:
        output.write("id,followers,friends" + '\n')
        output.flush()
        dataframe = pd.read_csv(input)
        ids = [int(elem) for elem in list(dataframe['id'])]
        for user in ids:
            followers = []
            friends = []
            try:
                for id in get_follower_ids(user):
                    followers.append(id)
            except:
                print(user)
            try:
                for id in get_friends_ids(user):
                    friends.append(id)
            except:
                print(user)
            follower_string = '"[' + ','.join(str(e) for e in followers) + ']"'
            friend_string = '"[' + ','.join(str(e) for e in friends) + ']"'
            output.write(str(user) + ',' + follower_string + ',' + friend_string + '\n')
            output.flush()


from ast import literal_eval


def create_network_graph():
    def add_node(graph, node, label, leaning):
        node = int(node)
        label = str(label)
        if not graph.has_node(node):
            graph.add_node(node, weight=1, label=label, leaning=leaning)
        else:
            graph.node[node]['weight'] += 1
            graph.node[node]['label'] = label
            graph.node[node]['leaning'] = leaning

    def add_edge(graph, node_1, label_1, leaning_1, node_2, label_2, leaning_2):
        node_1 = int(node_1)
        node_2 = int(node_2)

        label_1 = str(label_1)
        label_2 = str(label_2)

        leaning_1 = str(leaning_1)
        leaning_2 = str(leaning_2)

        if not graph.has_edge(node_1, node_2):
            if not graph.has_node(node_1):
                graph.add_node(node_1, weight= 1, label= label_1, leaning = leaning_1)
            if not graph.has_node(node_2):
                graph.add_node(node_2, weight= 1, label=  label_2, leaning =  leaning_2)
            graph.add_edge(node_1, node_2, weight = 1)
        else:
            graph[node_1][node_2]['weight'] += 1

    def write_graphml(graph, output_path):
        print("Writing Graph to " + str(output_path))
        nx.write_graphml(graph, output_path)

    def print_info(graph):
        list_of_degrees = [val for (node, val) in graph.degree()]
        print('--------------------------------------')
        print('Number of nodes: ' + str(graph.number_of_nodes()))
        print('Number of edges: ' + str(graph.number_of_edges()))
        print('Average degree: ' + str(
            round(float(sum(list_of_degrees)) / float(graph.number_of_nodes()), 2)))

        avg_sum = 0
        for _, data in graph.nodes(data=True):
            avg_sum += data['weight']

        print('Average weight: ' + str(round(float(avg_sum) / float(graph.number_of_nodes()), 2)))
        print('--------------------------------------')

    with open("./data/network_data.csv", 'r') as input, open("./data/political_leaning_with_ids.csv", 'r') as input_2:
        network_data = pd.read_csv(input, converters={'followers': literal_eval, 'friends': literal_eval})
        leaning_data = pd.read_csv(input_2)
        leaning_data.id = leaning_data.id.astype(int)
        leaning_data = leaning_data.set_index('id')

    graph = nx.Graph()
    for _, row in network_data.iterrows():
        a = row['id']
        name = leaning_data.at[a, 'user_name']
        leaning = leaning_data.at[a, 'leaning']
        add_node(graph, row['id'], name, leaning)
        for follower in row['followers']:
            add_edge(graph, row['id'], name, leaning, follower, str(follower), 'Unknown')
        for friend in row['friends']:
            add_edge(graph, row['id'], name, leaning, friend, str(friend), 'Unknown')

    print_info(graph)
    write_graphml(graph, "./data/test_graph.graphml")


def get_stats(threshold):
    with open("./data/follower_info_cleaned.csv", 'r') as input:
        df = pd.read_csv(input)
        a = 0
        b = 0
        for _, row in df.iterrows():
            a += row['followers_count']
            b += row['friends_count']
        print(a / 5000 * 15)
        print(b / 5000 * 15)
        print(a / 5000 * 15 + b / 5000 * 15)
        print((threshold * len(df) / 5000 * 15) / 60)


#get_followers()
create_network_graph()
