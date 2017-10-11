# -*- coding: utf-8 -*-

import pandas as pd
import collections as cl
import sys

sampling_count = 350
outfile_buy = 'dataset_buy_tensorflow' + str(sampling_count) + '.csv'
outfile_sell = 'dataset_sell_tensorflow' + str(sampling_count) + '.csv'

# def checkArg():

#     argv = sys.argv
#     argc = len(argv)
#     if (argc != 2):
#         print ('Usage: python %s arg1(azure/tensorflow)' %argv[0])
#         print ('Ex: python %s "azure"' %argv[0])
#         quit()
#     type = argv[1]
#     generateDataset(type)

def sampling():

    print ("processing...")
    df_buy = pd.read_csv("tmp_20170911/dataset_buy_tensorflow.csv")
    df_sell = pd.read_csv("tmp_20170911/dataset_sell_tensorflow.csv")

    up_sign = "0"
    down_sign = "1"
    stay_sign = "2"

    # 買借レート変動ごとdataframeを分割する
    df_buy_up = df_buy.query("fluctuation_buy_rate == @up_sign")
    df_buy_down = df_buy.query("fluctuation_buy_rate == @down_sign")
    df_buy_stay = df_buy.query("fluctuation_buy_rate == @stay_sign")

    # 売貸レート変動ごとdataframeを分割する
    df_sell_up = df_sell.query("fluctuation_sell_rate == @up_sign")
    df_sell_down = df_sell.query("fluctuation_sell_rate == @down_sign")
    df_sell_stay = df_sell.query("fluctuation_sell_rate == @stay_sign")

    print("buy count ...")
    print(len(df_buy_up.index))
    print(len(df_buy_down.index))
    print(len(df_buy_stay.index))
    print("sell count ...")
    print(len(df_sell_up.index))
    print(len(df_sell_down.index))
    print(len(df_sell_stay.index))

    print ("undersampling ...")

    # サンプリングする
    df_buy_up_sampled = df_buy_up.sample(n=sampling_count, replace=True)
    df_sell_up_sampled = df_sell_up.sample(n=sampling_count, replace=True)
    df_buy_stay_sampled = df_buy_stay.sample(n=sampling_count, replace=True)
    df_sell_stay_sampled = df_sell_stay.sample(n=sampling_count, replace=True)
    df_buy_down_sampled = df_buy_down.sample(n=sampling_count, replace=True)
    df_sell_down_sampled = df_sell_down.sample(n=sampling_count, replace=True)

    ### merge datasets
    df_buy_out = pd.concat([df_buy_up_sampled, df_buy_down_sampled, df_buy_stay_sampled])
    df_sell_out = pd.concat([df_sell_up_sampled, df_sell_down_sampled, df_sell_stay_sampled])

    ### ファイル出力
    df_buy_out.to_csv(outfile_buy)
    df_sell_out.to_csv(outfile_sell)
    print(len(df_buy_out.index))
    print(len(df_sell_out.index))


if __name__ == '__main__':
    sampling()
