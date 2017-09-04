
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

df2 = convert2number(df, 'mapping_4_stock_snm.txt', stock_snm_list_uq)
df3 = convert2number(df2, 'mapping_4_rate.txt', rate_fluctuation_list_uq)
df4 = convert2number(df3, 'mapping_4_date.txt', date_list_uq).dropna()

# df4dataset = df

# f = open('mapping_4_stock_snm.txt', 'w')
# mapping_4_stock_snm = cl.OrderedDict()
# num4stock_snm = 0
# for stock_snm in stock_snm_list_uq:
#     mapping_4_stock_snm[stock_snm] = num4stock_snm
#     df4dataset = df4dataset.replace(stock_snm, num4stock_snm)
#     num4stock_snm+=1
#     f.write(stock_snm + " " + str(mapping_4_stock_snm[stock_snm]) + "\n")
# f.close()

# f = open('mapping_4_rate.txt', 'w')
# mapping_4_rate_fluctuation = cl.OrderedDict()
# num4rate=0
# for rate_fluctuation in rate_fluctuation_list_uq:
#     mapping_4_rate_fluctuation[rate_fluctuation] = num4rate
#     df4dataset = df4dataset.replace(rate_fluctuation, num4rate)
#     num4rate+=1
#     f.write(rate_fluctuation + " " + str(mapping_4_rate_fluctuation[rate_fluctuation]) + "\n")
# f.close()

df4.to_csv('dataset4tensorflow.csv')


