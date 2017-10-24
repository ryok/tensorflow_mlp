import numpy as np
import tensorflow as tf
import csv
import os
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report
# import seaborn as sns


# prediction_type = 'buy'
prediction_type = 'sell'
# lambda_2 = 0.0006
lambda_2 = 0.00001
# learning_rate = 0.001
# learning_rate = 0.01
# learning_rate = 0.087
learning_rate = 0.07


if prediction_type == 'buy':
    read_file = 'tmp_20170911/dataset_buy_tensorflow.csv'
    # read_file = 'tmp_20170911/dataset_buy_tensorflow350.csv'
    supervised_label = 'fluctuation_buy_rate'
elif prediction_type == 'sell':
    read_file = 'tmp_20170911/dataset_sell_tensorflow.csv'
    # read_file = 'tmp_20170911/dataset_sell_tensorflow200.csv'
    supervised_label = 'fluctuation_sell_rate'
else:
    print('prediction type error...')
    exit()


### reset computational graph
ops.reset_default_graph()


if not os.path.exists(read_file):
    print ('can not find csv file...')
    exit()


### データをメモリに読み込む / read supply demand forecast data into memory
supply_demand_data = []
with open(read_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    supply_demand_header = next(csv_reader)
    for row in csv_reader:
        supply_demand_data.append(row)

supply_demand_data = [[float(x) for x in row] for row in supply_demand_data]


### 目的変数を抽出 / Extract y-target
df_dataset = pd.read_csv(read_file)
y_vals = np.array(df_dataset[supervised_label])
# y_vals = np.array([x[39] for x in supply_demand_data])
m = len(y_vals)
ydata = np.zeros((m, 3))
for i in range(m):
    ic = int(y_vals[i])
    ydata[i, ic] = 1.0


### 説明変数を抽出 / Filter for features of interest
cols_of_interest = ['stock_snm_x','contract_count_10','contract_count_11','contract_count_14','contract_count_15','contract_count_20','contract_count_21','contract_count_24','contract_count_25','contract_count_30','contract_count_31','contract_face_10','contract_face_11','contract_face_14','contract_face_15','contract_face_20','contract_face_21','contract_face_24','contract_face_25','contract_face_30','contract_face_31','zanzon','bid','simple_yield_fluctuation_2to1','simple_yield_fluctuation_3to2','amount_sum','balance']
x_vals = np.array([[x[ix] for ix, feature in enumerate(supply_demand_header) if feature in cols_of_interest] for x in supply_demand_data])


# set for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)


# Declare batch size
batch_size = 90
# batch_size = 1000
# batch_size = 120


### データセットをトレーニングセットとテストセットに分割 / Split data into train/test = 70%/30%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.7), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
# y_vals_train = y_vals[train_indices]
y_vals_train = ydata[train_indices]
# y_vals_test = y_vals[test_indices]
y_vals_test = ydata[test_indices]


### min-maxスケーリングを使って正則化 / Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


# Create graph
sess = tf.Session()


# Initialize placeholders
x_data = tf.placeholder(shape=[None, 27], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)


def init_variable(shape):
    return(tf.Variable(tf.random_normal(shape=shape)))

# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
    
def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)


### ロジスティック層の定義を作成 / Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    # We separate the activation at the end because the loss function will
    # implement the last sigmoid necessary
    if activation:
        return(tf.nn.sigmoid(linear_layer))
        # return(tf.nn.relu(linear_layer))
    else:
        return(linear_layer)
        # return(tf.nn.softmax(linear_layer))

# Create a fully connected layer:
# def fully_connected(input_layer, weights, biases):
#     layer = tf.add(tf.matmul(input_layer, weights), biases)
#     return(tf.nn.relu(layer))        

# def final_out(input_layer, weights, biases):
#     layer = tf.add(tf.matmul(input_layer, weights), biases)
#     # return(tf.nn.softmax(layer))
#     return layer


#--------Create the first layer (60 hidden nodes)--------
weight_1 = tf.Variable(tf.random_normal([27,60],mean=0.0,stddev=0.05))
# weight_1 = tf.Variable(tf.random_normal(shape=[27,60]))
bias_1 = tf.Variable(tf.random_normal(shape=[60]))
# bias_1 = tf.Variable(tf.zeros([60]))
layer_1 = logistic(x_data, weight_1, bias_1)
# layer_1 = fully_connected(x_data, weight_1, bias_1)

#--------Create second layer (60 hidden nodes)--------
weight_2 = tf.Variable(tf.random_normal([60,60],mean=0.0,stddev=0.05))
bias_2 = tf.Variable(tf.random_normal(shape=[60]))
layer_2 = logistic(layer_1, weight_2, bias_2)

#--------Create output layer (3 output value)--------
weight_4 = tf.Variable(tf.random_normal([60,3],mean=0.0,stddev=0.05))
# weight_4 = tf.Variable(tf.random_normal(shape=[60,3]))
bias_4 = tf.Variable(tf.random_normal(shape=[3]))
# bias_4 = tf.Variable(tf.zeros([3]))
final_output = logistic(layer_2, weight_4, bias_4, activation=False)

### 損失関数を作成 / Declare loss function (Cross Entropy loss)
# cross_entropy = -tf.reduce_sum(y_target*tf.log(final_output))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=final_output))
# loss = tf.reduce_mean(tf.abs(y_target - final_output))
# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target))
### 正則化 / Regularization terms (weight decay)
L2_sqr = tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(bias_1) + tf.nn.l2_loss(weight_2) + tf.nn.l2_loss(bias_2) + tf.nn.l2_loss(weight_4) + tf.nn.l2_loss(bias_4)
# lambda_2 = 0.0007
loss = cross_entropy + lambda_2 * L2_sqr
# loss = cross_entropy


### 最適化関数を作成 / Declare optimizer
my_opt = tf.train.AdamOptimizer(learning_rate)
# my_opt = tf.train.GradientDescentOptimizer(learning_rate)
# my_opt = tf.train.AdagradOptimizer(learning_rate)
# my_opt = tf.train.AdadeltaOptimizer(learning_rate)
# my_opt = tf.train.RMSPropOptimizer(learning_rate)
# my_opt = tf.train.MomentumOptimizer(learning_rate, 0)
train_step = my_opt.minimize(loss)


### Actual Prediction
# prediction = tf.round(final_output)
with tf.name_scope("test") as scope:
    correct_prediction = tf.cast(tf.equal(tf.argmax(final_output, 1), tf.argmax(y_target, 1)), tf.float32 )
    # correct_prediction = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    # matrix = tf.confusion_matrix(y_target, prediction)


### tensor board
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log', graph = sess.graph)


### confusion matrix
predictions = tf.argmax(final_output, 1)
# predictions = tf.argmax(prediction, 1)
actuals = tf.argmax(y_target, 1)
# tf_recall = tf.metrics.recall(actuals, predictions)
confmat = tf.confusion_matrix(actuals, predictions)


### 変数を初期化 / Initialize variables
init = tf.global_variables_initializer()
sess.run(init)


### Training loop
### 損失ベクトルと正解ベクトルを初期化
loss_vec = []
test_loss_vec = []
train_acc = []
test_acc = []
for i in range(3000):
    ### バッチを選択するためのインデックスをランダムに選択
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    ### ランダムな値でバッチを取得
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]

    ### トレーニングステップを実行
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    ### トレーニングセットの損失値を取得
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    ### トレーニングセットの正解率を取得
    train_result = sess.run([summary_op, accuracy], feed_dict={x_data: x_vals_train, y_target: y_vals_train})
    summary_str = train_result[0]
    temp_acc_train = train_result[1]
    train_acc.append(temp_acc_train)
    summary_writer.add_summary(summary_str, i)

    ### テストセットの正解率を取得
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_acc.append(temp_acc_test)

    ### テストセットの損失関数
    temp_loss_test = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_loss_vec.append(temp_loss_test)

    if (i+1)%150==0:
        print("step %d, training accuracy %g"%(i, temp_acc_train))
        print("step %d, test accuracy %g"%(i, temp_acc_test))
        print('Loss = ' + str(temp_loss))
        print('Test Loss = ' + str(temp_loss_test))


### 混合行列 / confusion matrix の表示
print('-----')
print(sess.run(
    confmat,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    )


### classification report出力
ac = sess.run(
    actuals,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})
pr = sess.run(
    predictions,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})
print(classification_report(ac, pr))


### seaborn.heatmap を使ってconfusion matrixをプロットする1
# fig, ax = plt.subplots(figsize = (2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
# plt.xlabel('ground_truth')
# plt.ylabel('prediction')
# plt.show()

### confusion matrixをプロットする2
# index = list("012")
# columns = list("012")
# df = pd.DataFrame(confmat, index=index, columns=columns)

# fig = plt.figure(figsize = (3,3))
# sns.heatmap(df, annot=True, square=True, fmt='.0f', cmap="Blues")
# plt.title('hand_written digit classification')
# plt.xlabel('ground_truth')
# plt.ylabel('prediction')
# fig.savefig("conf_mat.png")


### 損失値をプロット / Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

### 損失値をプロット / Plot loss over time
plt.plot(test_loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation (Test)')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

### トレーニングセットとテストセットの正解率をプロット / Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

