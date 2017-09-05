
import pandas as pd

df4 = pd.read_csv("input/datasets.csv")

print ("processing...")

# buy
df_buy_up = df4.query("fluctuation_buy_rate == 'up'")
df_buy_down = df4.query("fluctuation_buy_rate == 'down'")
df_buy_stay = df4.query("fluctuation_buy_rate == 'stay'")
print(len(df_buy_up))
print(len(df_buy_down))
print(len(df_buy_stay))

# buy
df_sell_up = df4.query("fluctuation_sell_rate == 'up'")
df_sell_down = df4.query("fluctuation_sell_rate == 'down'")
df_sell_stay = df4.query("fluctuation_sell_rate == 'stay'")
print(len(df_sell_up.index))
print(len(df_sell_down.index))
print(len(df_sell_stay.index))

# stayのデータをアンダーサンプリングする
df_buy_stay_sampled = df_buy_stay.sample(n=220, replace=True)
df_sell_stay_sampled = df_sell_stay.sample(n=220, replace=True)
# print(len(df_stay_sampled))

### merge datasets
df_buy = pd.concat([df_buy_up, df_buy_down, df_buy_stay_sampled])
df_sell = pd.concat([df_sell_up, df_sell_down, df_sell_stay_sampled])

df_buy.to_csv('dataset_buy_azureml.csv')
df_sell.to_csv('dataset_sell_azureml.csv')
print(len(df_buy.index))
print(len(df_sell.index))
