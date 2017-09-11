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
import seaborn as sns


# reset computational graph
ops.reset_default_graph()

read_file = 'dataset_buy_tensorflow.csv'
# read_file = 'dataset_sell_tensorflow.csv'

if not os.path.exists(read_file):
    print ('can not find csv file...')
    exit()

# read supply demand forecast data into memory
supply_demand_data = []
with open(read_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    supply_demand_header = next(csv_reader)
    for row in csv_reader:
        supply_demand_data.append(row)

supply_demand_data = [[float(x) for x in row] for row in supply_demand_data]

df_dataset = pd.read_csv(read_file)
y_vals = np.array(df_dataset['fluctuation_buy_rate'])

# Extract y-target
# y_vals = np.array([x[39] for x in supply_demand_data])
m = len(y_vals)
ydata = np.zeros((m, 3))
for i in range(m):
    ic = int(y_vals[i])
    ydata[i, ic] = 1.0

print(ydata)

# Filter for features of interest
cols_of_interest = ['stock_snm_x','contract_count_10','contract_count_11','contract_count_14','contract_count_15','contract_count_20','contract_count_21','contract_count_24','contract_count_25','contract_count_30','contract_count_31','contract_face_10','contract_face_11','contract_face_14','contract_face_15','contract_face_20','contract_face_21','contract_face_24','contract_face_25','contract_face_30','contract_face_31','zanzon','bid','simple_yield_fluctuation_2to1','simple_yield_fluctuation_3to2','amount_sum','balance']
x_vals = np.array([[x[ix] for ix, feature in enumerate(supply_demand_header) if feature in cols_of_interest] for x in supply_demand_data])

# set for reproducible results
seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

# Declare batch size
batch_size = 90

# Split data into train/test = 70%/30%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.7), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
# y_vals_train = y_vals[train_indices]
y_vals_train = ydata[train_indices]
# y_vals_test = y_vals[test_indices]
y_vals_test = ydata[test_indices]


# Normalize by column (min-max norm)
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


# Create variable definition
def init_variable(shape):
    return(tf.Variable(tf.random_normal(shape=shape)))


# Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    # We separate the activation at the end because the loss function will
    # implement the last sigmoid necessary
    if activation:
        return(tf.nn.sigmoid(linear_layer))
    else:
        return(linear_layer)
        # return(tf.nn.softmax(linear_layer))
        

# First logistic layer (7 inputs to 7 hidden nodes)
A1 = init_variable(shape=[27,60])
b1 = init_variable(shape=[60])
logistic_layer1 = logistic(x_data, A1, b1)

# Final output layer (5 hidden nodes to 1 output)
A3 = init_variable(shape=[60,3])
b3 = init_variable(shape=[3])
final_output = logistic(logistic_layer1, A3, b3, activation=False)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target))
# cross_entropy = -tf.reduce_sum(y_target*tf.log(final_output))
# Regularization terms (weight decay)   
# L2_sqr = tf.nn.l2_loss(A1) + tf.nn.l2_loss(A3)
# lambda_2 = 0.01
# loss = cross_entropy + lambda_2 * L2_sqr

# Declare optimizer
my_opt = tf.train.AdamOptimizer(learning_rate = 0.002)
# my_opt = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train_step = my_opt.minimize(loss)



# Actual Prediction
prediction = tf.round(tf.nn.sigmoid(final_output))
# correct_prediction = tf.cast(tf.equal(prediction, y_target), tf.float32)
# accuracy = tf.reduce_mean(correct_prediction)

with tf.name_scope("test") as scope:
    correct_prediction = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y_target, 1)), tf.float32 )
    # correct_prediction = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    # matrix = tf.confusion_matrix(y_target, prediction)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log', graph = sess.graph)



predictions = tf.argmax(prediction, 1)
actuals = tf.argmax(y_target, 1)
tf_recall = tf.metrics.recall(actuals, predictions)
confusion_matrix = tf.confusion_matrix(actuals, predictions)
# cl_report = classification_report(actuals, predictions)



# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # rand_index = np.random.choice(batch_size, batch_size)
    rand_x = x_vals_train[rand_index]
    # rand_y = np.transpose([y_vals_train[rand_index]])
    rand_y = y_vals_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    train_result = sess.run([summary_op, accuracy], feed_dict={x_data: x_vals_train, y_target: y_vals_train})
    summary_str = train_result[0]
    temp_acc_train = train_result[1]
    train_acc.append(temp_acc_train)
    summary_writer.add_summary(summary_str, i)

    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_acc.append(temp_acc_test)

    if (i+1)%150==0:
        print("step %d, training accuracy %g"%(i, temp_acc_train))
        print("step %d, test accuracy %g"%(i, temp_acc_test))
        print('Loss = ' + str(temp_loss))
        

# precision recall
# print('Precision: %.3f' % precision_score(tf.argmax(y_target, 1), tf.argmax(prediction, 1)))
# print('Precision: %.3f' % precision_score(y_vals, prediction))


'''与えられたネットワークの正解率などを出力する。
'''

ones_like_actuals = tf.ones_like(actuals)
zeros_like_actuals = tf.zeros_like(actuals)
ones_like_predictions = tf.ones_like(predictions)
zeros_like_predictions = tf.zeros_like(predictions)

tp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, ones_like_predictions)
        ),
        "float"
    )
)

tn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
    )
)

fp_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, zeros_like_actuals),
            tf.equal(predictions, ones_like_predictions)
        ),
        "float"
    )
)

fn_op = tf.reduce_sum(
    tf.cast(
        tf.logical_and(
            tf.equal(actuals, ones_like_actuals),
            tf.equal(predictions, zeros_like_predictions)
        ),
        "float"
    )
)

tp, tn, fp, fn = sess.run(
    [tp_op, tn_op, fp_op, fn_op],
    feed_dict={x_data: x_vals_test, y_target: y_vals_test}
)

tpr = float(tp)/(float(tp) + float(fn))
fpr = float(fp)/(float(tp) + float(fn))

accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

recall = tpr
if (float(tp) + float(fp)):
    precision = float(tp)/(float(tp) + float(fp))
    f1_score = (2 * (precision * recall)) / (precision + recall)
else:
    precision = 0
    f1_score = 0

print('-----')
print('Precision = ', precision)
print('Recall = ', recall)
print('F1 Score = ', f1_score)
print('Accuracy = ', accuracy)

print('-----')
# con_matrix = sess.run(,
#     feed_dict={x_data: x_vals_test, y_target: y_vals_test}
# )
print(sess.run(
    confusion_matrix,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    )
# print(sess.run(
#     tf_recall,
#     feed_dict={x_data: x_vals_test, y_target: y_vals_test})
#     )

ac = sess.run(
    actuals,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})
pr = sess.run(
    predictions,
    feed_dict={x_data: x_vals_test, y_target: y_vals_test})

print(classification_report(ac, pr))
# print(cl_report)


# seaborn.heatmap を使ってプロットする
# index = list("012")
# columns = list("012")
# df = pd.DataFrame(confusion_matrix, index=index, columns=columns)

# fig = plt.figure(figsize = (3,3))
# sns.heatmap(df, annot=True, square=True, fmt='.0f', cmap="Blues")
# plt.title('hand_written digit classification')
# plt.xlabel('ground_truth')
# plt.ylabel('prediction')
# fig.savefig("conf_mat.png")

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

