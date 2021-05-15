from tensorflow.keras.layers import Input,ZeroPadding2D,Conv2D,MaxPool2D,Dropout,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



width = 28
height = 28

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000, width, height, 1).astype('float32')/255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32')/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=[28,28,1])

#Layer_1
H1_Zero = ZeroPadding2D(padding=(1,1))(inputs)
H1_Conv = Conv2D(64,kernel_size=3,strides=1,activation='relu')(H1_Zero)

#Layer_2
H2_Zero = ZeroPadding2D(padding=(1,1))(H1_Conv)
H2_Conv = Conv2D(64,kernel_size=3,strides=1,activation='relu')(H2_Zero)
H2_Pool = MaxPool2D(pool_size=(2,2),strides=2)(H2_Conv)

#Layer_3
H3_Zero1 = ZeroPadding2D(padding=(1,1))(H2_Pool)
H3_Conv1 = Conv2D(128,kernel_size=3,strides=1,activation='relu')(H3_Zero1)
H3_Zero2 = ZeroPadding2D(padding=(1,1))(H3_Conv1)
H3_Conv2 = Conv2D(128,kernel_size=3,strides=1,activation='relu')(H3_Zero2)
H3_Pool = MaxPool2D(pool_size=(2,2),strides=2)(H3_Conv2)

#Layer_4
H4_Zero1 = ZeroPadding2D(padding=(1,1))(H3_Pool)
H4_Conv1 = Conv2D(256,kernel_size=3,strides=1,activation='relu')(H4_Zero1)
H4_Zero2 = ZeroPadding2D(padding=(1,1))(H4_Conv1)
H4_Conv2 = Conv2D(256,kernel_size=3,strides=1,activation='relu')(H4_Zero2)
H4_Zero3 = ZeroPadding2D(padding=(1,1))(H4_Conv2)
H4_Conv3 = Conv2D(256,kernel_size=3,strides=1,activation='relu')(H4_Zero3)
H4_Pool = MaxPool2D(pool_size=(2,2),strides=2)(H4_Conv3)

#Layer_5
H5_Zero1 = ZeroPadding2D(padding=(1,1))(H4_Pool)
H5_Conv1 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H5_Zero1)
H5_Zero2 = ZeroPadding2D(padding=(1,1))(H5_Conv1)
H5_Conv2 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H5_Zero2)
H5_Zero3 = ZeroPadding2D(padding=(1,1))(H5_Conv2)
H5_Conv3 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H5_Zero3)
H5_Pool = MaxPool2D(pool_size=(2,2),strides=2)(H5_Conv3)

#Layer_6
H6_Zero1 = ZeroPadding2D(padding=(1,1))(H5_Pool)
H6_Conv1 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H6_Zero1)
H6_Zero2 = ZeroPadding2D(padding=(1,1))(H6_Conv1)
H6_Conv2 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H6_Zero2)
H6_Zero3 = ZeroPadding2D(padding=(1,1))(H6_Conv2)
H6_Conv3 = Conv2D(512,kernel_size=3,strides=1,activation='relu')(H6_Zero3)

# Layer_7
H7_Flat = Flatten()(H6_Conv3)

#Layer_8
H8_Full = Dense(4096,activation='relu')(H7_Flat)
H8_Drop = Dropout(0.5)(H8_Full)

#Layer_9
H8_Full = Dense(1000,activation='relu')(H8_Full)
H8_Drop = Dropout(0.5)(H8_Full)

#Layer_10
outputs = Dense(10,activation='softmax')(H8_Drop)

# model_make
model = Model(inputs,outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# practice
model.fit(x_train,y_train,batch_size=540,epochs=10)

# test code
score = model.evaluate(x_test,y_test) #[loss,acc]
print("loss : ",score[0])
print("acc : ",score[1])
