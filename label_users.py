import pandas as pd
import operator

input_path = './data_2/clean_data/election_debate_2016.csv'
input_path_l = './data_2/clean_data/election_debate_leaning_2016.csv'


with open(input_path) as input_data:
    tweets_df = pd.read_csv(input_data)


tweets_df['leaning'] = 'Unknown'
tweets_df = tweets_df.set_index('u_id',drop=False)


t = tweets_df.hashtags.apply(lambda x: ('CrookedHillary' in x or 'MAGA' in x or 'LockHerUp' in x or 'AmericaFirst' in x
                                        or 'DrainTheSwamp' in x or 'TrumpTrain' in x or 'Women4Trump' in x or 'VoteTrump' in x)
                                       and ('nevertrump' not in x) and ('DumpTrump' not in x) and ('NeverTrump' not in x)
                             and ('NotMyPresident' not in x))

df_temp = tweets_df[t]
for _,row in df_temp.iterrows():
    user = row['u_id']
    tweets_df.at[user,'leaning'] = "R"

p = tweets_df.hashtags.apply(lambda x: ('hillaryforpresident' in x or 'imwithher' in x or 'hillary' in x or 'NotMyPresident' in x
                                        or 'ImWithHer' in x or 'VOTEHILLARY' in x or 'VoteHILLARY' in x or 'VoteHillaryClinton' in x)
                                       and ('CrookedHillary' not in x) and ('MAGA' not in x) and ('AmericaFirst' not in x)
                             and ('LockHerUp' not in x))

df_temp = tweets_df[p]
for _,row in df_temp.iterrows():
    user = row['u_id']
    what = tweets_df.at[user, 'leaning']
    if row['leaning'] == 'R':
        print("Ambigous classification...")
        tweets_df.at[user, 'leaning'] = 'Unknown'
    else:
        tweets_df.at[user,'leaning'] = "D"


tweets_df.to_csv('./data_2/clean_data/election_debate_leaning_2016.csv')








