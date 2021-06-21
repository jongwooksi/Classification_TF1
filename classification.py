import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import random
os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.set_random_seed(777)
np.random.seed(777)


len_sequence = 1
inputcount = 48

epoch_num =10000
learning_rate = 0.0000002

def model(X, Y, trainX, trainY):
 
    W1 = tf.Variable(tf.random_normal([inputcount, 64], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([1]))
    L1 = tf.nn.relu(tf.matmul(X, W1)+ b1)
    L1 = tf.nn.dropout(L1, 0.8)

    W2 = tf.Variable(tf.random_normal([64, 1], stddev=0.01))

        
    output = tf.sigmoid(tf.matmul(L1, W2))
    
    return output
			
def nomalization_train(x):
    ary = np.asarray(x)
    ary = ( x - x.min(0))/(x.ptp(0)+ 1e-8 )
    return (ary - ary.min(axis=0)) / (ary.max(axis=0) - ary.min(axis=0) + 1e-8) 
    
def nomalization_test(x, x2):
    ary = np.asarray(x)
    ary2 = np.asarray(x2)
    ary = ( x - x.min(0))/(x.ptp(0)+ 1e-8 )
    return (ary2 - ary.min(axis=0)) / (ary.max(axis=0) - ary.min(axis=0) + 1e-8) 
    

def modifyExcel(raw_dataframe, flag):

    delHeader = ['Unnamed: 0','Input','DTYING','CTACTIONTEMPP','CTACTIONTEMPS', 'CTOUTTEMP', 'CTPRESSURE', 'HP1ACTIONTEMPP', 'HP1ACTIONTEMPS', 'HP1ENDTEMP',
                'HP2ACTIONTEMPP', 'HP2ACTIONTEMPS', 'HP2ENDTEMP', 'SPHUMIDITY', 'SPTEMP','WH1ACTIONTEMPP', 'WH1ACTIONTEMPS','WH1INFLUX', 'WH1INTEMP', 'WH1OUTTEMP', 
                'WH2ACTIONTEMPP','WH2ACTIONTEMPS', 'WH2INFLUX', 'WH2INTEMP', 'WH2OUTTEMP','WH3ACTIONTEMPP', 'WH3ACTIONTEMPS', 'WH3INFLUX', 'WH3INTEMP','WH3OUTTEMP', 
                'WH4ACTIONTEMPP', 'WH4ACTIONTEMPS', 'WH4INFLUX', 'WH4INTEMP', 'WH4OUTTEMP', 'DATE', 'LOT','InjSpeed', 'RecovPos', 'paintTemp', 'SprayS', 'PatternS',
                'HPTransPos', 'ApplyP','BackPress', 'LABORATOR', 'CycleTime']

    for headerName in delHeader:
        del raw_dataframe[headerName]

    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('GOOD', 1)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('Foreign', 0)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('Gas', 0)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('Pollution', 0)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('Scratch', 0)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('Unmolded', 0)
    raw_dataframe['RESULT'] = raw_dataframe['RESULT'].replace('LargeCut', 0)
    
    aa = []

    if flag is True:
        count = 0
        
        for i in range(1, 2988):
            temp = raw_dataframe.values[i][-1]
            
            if temp == 0.0:
                for k in range(1,34):
                    for j in range(inputcount-30):
                        raw_dataframe.values[i][j] = raw_dataframe.values[i][j]+ random.random()

                    raw_dataframe.loc[2987+count+k]= raw_dataframe.values[i]
                    raw_dataframe.values[2987+count+k][-1] = 0
                   
                count += 33


    return raw_dataframe



raw_dataframe = pd.read_excel('./train.xlsx', engine='openpyxl') 
raw_dataframe = modifyExcel(raw_dataframe, True)
raw_dataframe.info() 


inform = raw_dataframe.values[:].astype(np.float) 

x = nomalization_train(inform[:, :-1])
y = (inform[:, -1:])

dataX = [] 
dataY = [] 
 
for i in range(0, len(y)):
    _x = x[i : i+len_sequence]
    _y = y[i] 
    dataX.append(_x) 
    dataY.append(_y)
    

train_size = int(len(dataY) * 0.1)

test_size = len(dataY) - train_size

x_train = np.array(dataX[train_size:len(dataX)-train_size])[:,0]
y_train = np.array(dataY[train_size:len(dataY)-train_size])

testX_1 = np.array(dataX[0:train_size])[:,0]
testX_2 = np.array(dataX[len(dataX)-train_size:len(dataX)])[:,0]
testX = np.concatenate([testX_1, testX_2], axis=0)


testY_1 = np.array(dataY[0:train_size])
testY_2 = np.array(dataY[len(dataY)-train_size:len(dataY)])
testY = np.concatenate([testY_1, testY_2], axis=0)

X = tf.placeholder(tf.float32, [None, inputcount])
Y = tf.placeholder(tf.float32, [None, 1])

hypo = model(X, Y, x_train, y_train)
loss = tf.reduce_mean(tf.square(hypo - Y))

optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])

prediction = tf.cast(hypo>0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, prediction), dtype=tf.float32))
 
test_predict = ''        
 
train_error_save = []
test_error_save = []
mse = tf.reduce_mean(tf.squared_difference(targets, predictions))
 


raw_dataframe2 = pd.read_excel('./test.xlsx', engine='openpyxl') 
raw_dataframe2 = modifyExcel(raw_dataframe2, False)
raw_dataframe2.info() 


inform2 = raw_dataframe2.values[:].astype(np.float) 
print(inform2)


x2 = nomalization_test(x, inform2[:, :-1])
x2[x2>1] = 1
x2[x2<0] = 0
y2 = (inform2[:, -1:])


sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: x_train, Y: y_train})

    if ((epoch+1) % 1000 == 0) or (epoch == epoch_num-1): 
        
        train_predict = sess.run(hypo, feed_dict={X: x_train})
        train_error = sess.run(mse, feed_dict={targets: y_train, predictions: train_predict})
        
        test_predict = sess.run(hypo, feed_dict={X: testX})
        test_error = sess.run(mse, feed_dict={targets: testY, predictions: test_predict})
        
        train_error_save.append(train_error)
        test_error_save.append(test_error)
        
        print("epoch: {}, loss: {:.4f} train_error(A): {:4f}, test_error(B): {:4f}".format(epoch+1, _loss, train_error, test_error))
        
    if (epoch+1)%1000 == 0 :
        h, p, a = sess.run([hypo, prediction, accuracy], feed_dict={X: x2, Y: y2})


        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for pred,label,k in zip(p, y2, h):
            if label[0] == 1:
                if pred[0] == 1:
                    tp += 1
                else:
                    fn += 1

            else:
                if pred[0] == 0:
                    tn += 1
                else:
                    fp += 1
        
        print("TP: {:.4f}, TN: {:.4f}, FP: {:.4f}, FN: {:.4f}".format(tp, tn, fp, fn))

        if tp == 0 and fp == 0 :
            fp = 1

        if tn == 0 and fn == 0 :
            fn = 1

        recall = (tp / (tp + fn) + tn / (tn + fp))/2
        precision = (tp / (tp + fp) + tn / (tn + fn))/2
        f1_score = 2*recall*precision / (recall + precision)


        print("Aveage F1     {:.4f}%".format(f1_score * 100))
        print("Aveage recall {:.4f}%".format(recall * 100))



