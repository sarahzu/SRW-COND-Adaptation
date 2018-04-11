import pandas as pd

input_path = "/Users/lenzbaumann/Downloads/twitter_data.csv"

df_twitter_data = pd.read_csv(open(input_path))

#print(df_twitter_data.columns.values)

df_high_comf = df_twitter_data[df_twitter_data['gender:confidence']==1]
#for i, row in df_twitter_data.iterrows():
 #  print(row)

print(len(df_twitter_data))
print(len(df_high_comf))
