import numpy as np
import tensorflow as tf
import csv
import os
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
# from sklearn.feature_extraction import DictVectorizer
import pandas as pd

# reset computational graph
ops.reset_default_graph()

# read_file = 'input/easy_datasets_buy3.csv'
read_file = 'dataset4tensorflow.csv'

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
# df = pd.read_csv("dataset4tensorflow.csv")
# d = df[["fluctuation_sell_rate"]].to_dict('record')
# Extract y-target
y_vals = np.array([x[39] for x in supply_demand_data])
m = len(y_vals)
ydata = np.zeros((m, 3))
for i in range(m):
    ic = int(y_vals[i])
    ydata[i, ic] = 1.0

print(ydata)

# Filter for features of interest
cols_of_interest = ['contract_count_10','contract_count_11','contract_count_14','contract_count_15','contract_count_20','contract_count_21','contract_count_24','contract_count_25','contract_count_30','contract_count_31','contract_face_10','contract_face_11','contract_face_14','contract_face_15','contract_face_20','contract_face_21','contract_face_24','contract_face_25','contract_face_30','contract_face_31','zanzon','bid','simple_yield_1days_from_ope','simple_yield_2days_from_ope','simple_yield_fluctuation_2to1','simple_yield_fluctuation_3to2','amount_sum','balance']
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
x_data = tf.placeholder(shape=[None, 28], dtype=tf.float32)
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
A1 = init_variable(shape=[28,60])
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
# my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# TensorBoard
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('log', graph = sess.graph)

# Actual Prediction
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
# with tf.name_scope('accuracy'):
#   with tf.name_scope('correct_prediction'):
#     correct_prediction = tf.cast(tf.equal(prediction, y_target), tf.float32)
#   with tf.name_scope('accuracy'):
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.summary.scalar('accuracy', accuracy)

# TensorBoard
# logloss = tf.scalar_summary('loss_w_L2', loss)
# logaccu = tf.scalar_summary('accuracy', accuracy)
# summary_op = tf.merge_summary([logloss, logaccu])

# Training loop
loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
    # rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_index = np.random.choice(batch_size, batch_size)
    rand_x = x_vals_train[rand_index]
    # rand_y = np.transpose([y_vals_train[rand_index]])
    rand_y = y_vals_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: y_vals_train})
    # temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    # temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)

    if (i+1)%150==0:
        print("step %d, training accuracy %g"%(i, temp_acc_train))
        print("step %d, test accuracy %g"%(i, temp_acc_test))
        print('Loss = ' + str(temp_loss))
        
        
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
