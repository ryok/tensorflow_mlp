
import pandas as pd
import collections as cl

df = pd.read_csv("input/datasets.csv")

stock_snm_list = df["stock_snm_x"]
rate_fluctuation_list = df["fluctuation_sell_rate"]
stock_snm_list_uq = list(set(stock_snm_list))
rate_fluctuation_list_uq = list(set(rate_fluctuation_list))
date_list = df["date"]
date_list_uq = list(set(date_list))

print ("processing...")
# print(stock_snm_list_uq)

def convert2number(df4dataset, file_name, target_list):
    f = open(file_name, 'w')
    mapping_dict = cl.OrderedDict()
    number = 0
    for item in target_list:
        mapping_dict[item] = number
        df4dataset = df4dataset.replace(item, number)
        number+=1
        f.write(item + " " + str(mapping_dict[item]) + "\n")
    f.close()
    return df4dataset

df2 = convert2number(df, 'mapping_result_stock_snm.txt', stock_snm_list_uq)
df3 = convert2number(df2, 'mapping_result_rate.txt', rate_fluctuation_list_uq)
df4 = convert2number(df3, 'mapping_result_date.txt', date_list_uq).dropna()

# buy
df_buy_up = df4.query("fluctuation_buy_rate == 0")
df_buy_down = df4.query("fluctuation_buy_rate == 1")
df_buy_stay = df4.query("fluctuation_buy_rate == 2")
print(len(df_buy_up))
print(len(df_buy_down))
print(len(df_buy_stay))

# buy
df_sell_up = df4.query("fluctuation_sell_rate == 0")
df_sell_down = df4.query("fluctuation_sell_rate == 1")
df_sell_stay = df4.query("fluctuation_sell_rate == 2")
print(len(df_sell_up.index))
print(len(df_sell_down.index))
print(len(df_sell_stay.index))

df_buy_stay_sampled = df_buy_stay.sample(n=220, replace=True)
df_sell_stay_sampled = df_sell_stay.sample(n=220, replace=True)
# print(len(df_stay_sampled))

### merge datasets
df_buy = pd.concat([df_buy_up, df_buy_down, df_buy_stay_sampled])
df_sell = pd.concat([df_sell_up, df_sell_down, df_sell_stay_sampled])

df_buy.to_csv('dataset_buy_tensorflow.csv')
df_sell.to_csv('dataset_sell_tensorflow.csv')
print(len(df_buy.index))
print(len(df_sell.index))


