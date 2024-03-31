import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Input,Dense,Dropout, Conv1D, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import hamming_loss
from keras.callbacks import ModelCheckpoint

encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
preprocessing_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
BATCH_SIZE = 64

#  METRICS LIKE ACCURACY AND HAMMING LOSS 
 
def metrics(y_actual,y_pred):
    print("hamming loss = {}".format(hamming_loss(y_actual, y_pred)))
    print("accuracy score = {}".format(accuracy_score(y_actual, y_pred)))

    print([f1_score(y_actual.iloc[:,count],y_pred[:,count], average='weighted') for count in range(len(categories))])
    print([accuracy_score(y_actual.iloc[:,count],y_pred[:,count]) for count in range(len(categories))])
    print([precision_score(y_actual.iloc[:,count],y_pred[:,count]) for count in range(len(categories))])
    print([recall_score(y_actual.iloc[:,count],y_pred[:,count]) for count in range(len(categories))])

if __name__ == '__main__':
    # LOADING DATASET
    
    df_train= pd.read_csv('jigsaw-toxic-comment-train.csv')
    print(df_train.head(10))
    
    #  SPITTING OF DATASET
    
    categories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    X_train = df_train['comment_text']
    y_train = df_train[categories]
    # CREATING MODEL AND LOADING CHECKPOINTS
    
    ###loading bert model 
    preprocessing_model = hub.KerasLayer(preprocessing_url)
    bert_model = hub.KerasLayer(encoder_url)
    
    # Neural Network
    #bert model
    text_input = Input(shape = (),dtype=tf.string,name = 'text')
    preprocessed_text = preprocessing_model(text_input)
    output = bert_model(preprocessed_text)

    # ANN layers
    l = Conv1D (32,2,name = 'Conv1d_1')(output['sequence_output'])
    l = Conv1D (64,2,name = 'Conv1d_2')(l)
    l = GlobalMaxPooling1D(name = 'global_max_pool_1')(l)
    l = Dense(512,name = 'dense_l')(l)
    l = Dropout(0.3,name = 'dropout_2')(l)

    l = Dense(6,activation = 'sigmoid',name = 'output')(l)

    model2 = tf.keras.Model(inputs = [text_input],outputs = [l])

    model2 = tf.keras.Model(inputs = [text_input],outputs = [l])
    print(model2.summary())
    
    opt = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
    Metrics = [
        tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'),
        tf.keras.metrics.Precision(name = 'precision'),
        tf.keras.metrics.Recall(name = 'recall')
    ]
    
    ### Code for checkpoint saving
    filepath= "bert_ANN_model_full_checkpoint-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath)
    callbacks_list = [checkpoint]
    
    model2.compile(optimizer = opt,loss = 'binary_crossentropy',metrics = Metrics)
    # model2.load_weights('bert_ANN_model_checkpoint-43.h5')
    # model2.fit(X_train,y_train,epochs = 7,batch_size = BATCH_SIZE,callbacks = callbacks_list)
    
    model2.load_weights('bert_ANN_full_model.h5')
    
    df_test_y = pd.read_csv('test_label.csv')
    df_test = pd.read_csv('test.csv')
    y_pred = model2.predict(df_test['content'])[:,0]
    for j in range(len(y_pred)):
            if y_pred[j] not in [0,1]:
                if y_pred[j] > 0.5:
                    y_pred[j] = 1
                else:
                    y_pred[j] = 0

    metrics(df_test_y['toxic'],y_pred)