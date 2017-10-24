# -*- coding: utf-8 -*-

import pandas as pd
import collections as cl
import sys

def checkArg():

    argv = sys.argv
    argc = len(argv)
    if (argc != 2):
        print ('Usage: python %s arg1(azure/tensorflow)' %argv[0])
        print ('Ex: python %s "azure"' %argv[0])
        quit()
    type = argv[1]
    generateDataset(type)


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


def generateDataset(type):

    print ("processing...")
    df = pd.read_csv("input/datasets.csv")

    if type == "tensorflow":
        
        print ("converting string to number ...")
        ### tensorflowの場合、文字列を実数に置き換える
        ### 実数と文字列のマッピング結果をファイル出力する
        stock_snm_list = df["stock_snm_x"]
        rate_fluctuation_list = df["fluctuation_sell_rate"]
        stock_snm_list_uq = list(set(stock_snm_list))
        # rate_fluctuation_list_uq = list(set(rate_fluctuation_list))
        rate_fluctuation_list_uq = ["up", "down", "stay"]
        date_list = df["date"]
        date_list_uq = list(set(date_list))
        df2 = convert2number(df, 'mapping_result_stock_snm.txt', stock_snm_list_uq)
        df3 = convert2number(df2, 'mapping_result_rate.txt', rate_fluctuation_list_uq)
        df4 = convert2number(df3, 'mapping_result_date.txt', date_list_uq).dropna()

        up_sign = "0"
        down_sign = "1"
        stay_sign = "2"
        outfile_buy = 'dataset_buy_tensorflow.csv'
        outfile_sell = 'dataset_sell_tensorflow.csv'

    elif type == "azure":

        df4 = df
        up_sign = "up"
        down_sign = "down"
        stay_sign = "stay"

        outfile_buy = 'dataset_buy_azureml.csv'
        outfile_sell = 'dataset_sell_azureml.csv'

    else:
        print("Unexpected argument. ")
        exit()

    # 買借レート変動ごとdataframeを分割する
    df_buy_up = df4.query("fluctuation_buy_rate == @up_sign")
    df_buy_down = df4.query("fluctuation_buy_rate == @down_sign")
    df_buy_stay = df4.query("fluctuation_buy_rate == @stay_sign")

    # 売貸レート変動ごとdataframeを分割する
    df_sell_up = df4.query("fluctuation_sell_rate == @up_sign")
    df_sell_down = df4.query("fluctuation_sell_rate == @down_sign")
    df_sell_stay = df4.query("fluctuation_sell_rate == @stay_sign")

    print("buy count ...")
    print(len(df_buy_up.index))
    print(len(df_buy_down.index))
    print(len(df_buy_stay.index))
    print("sell count ...")
    print(len(df_sell_up.index))
    print(len(df_sell_down.index))
    print(len(df_sell_stay.index))

    print ("undersampling ...")

    # stayのデータをアンダーサンプリングする
    df_buy_stay_sampled = df_buy_stay.sample(n=500, replace=True)
    df_sell_stay_sampled = df_sell_stay.sample(n=500, replace=True)

    ### merge datasets
    df_buy = pd.concat([df_buy_up, df_buy_down, df_buy_stay_sampled])
    df_sell = pd.concat([df_sell_up, df_sell_down, df_sell_stay_sampled])

    ### ファイル出力
    df_buy.to_csv(outfile_buy)
    df_sell.to_csv(outfile_sell)
    print(len(df_buy.index))
    print(len(df_sell.index))


if __name__ == '__main__':
    checkArg()
