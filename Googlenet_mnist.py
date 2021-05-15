# outmemory난다.
from tensorflow.keras import models  # compile 형성 위함
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Dropout, Flatten, Input,
                                     MaxPool2D, concatenate)
from tensorflow.keras.models import Model  # input, output 설정
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, width, height, 1).astype('float32')/255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32')/255.0

num_classes = 10
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 함수는 동사형식으로(매개변수는 명사로)
def Inception(input_layer,ch_num):
    conv1 = Conv2D(ch_num,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
    conv2 = Conv2D(ch_num,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
    conv2 = Conv2D(ch_num,kernel_size=(3,3),padding='same',activation='relu')(conv2)
    conv3 = Conv2D(ch_num,kernel_size=(1,1),padding='same',activation='relu')(input_layer)
    conv3 = Conv2D(ch_num,kernel_size=(5,5),padding='same',activation='relu')(conv3)
    root4 = MaxPool2D(pool_size=(3,3),strides=1,padding='same')(input_layer)
    root4 = Conv2D(ch_num,kernel_size=(1,1),padding='same',activation='relu')(root4)

    output = concatenate([conv1,conv2,conv3,root4], axis=-1) # c소문자다!,그냥 결과를 줄줄이 붙여준다.
    return output

inputs = Input(shape=(28,28,1))

# layer_1
L1 = Conv2D(64,kernel_size=(7,7),strides=2,padding='same',activation='relu')(inputs)
L1 = BatchNormalization()(L1)
L1 = Conv2D(192,kernel_size=(3,3),strides=1,padding='same',activation='relu')(L1)
L1 = BatchNormalization()(L1)

# layer_2
L2 = Inception(L1,256)
L2 = Inception(L2,480)
L2 = MaxPool2D(pool_size=(3,3),strides=2,padding='same')(L2)

# layer_3
L3 = Inception(L2,512)

# sublayer_1
sub1 = Flatten()(L3)
sub1 = Dense(512,activation='relu')(sub1)
sub1 = Dense(10,activation='relu')(sub1)

# layer_3 (continue)
L3 = Inception(L3,512)
L3 = Inception(L3,512)
L3 = Inception(L3,528)

# sublayer_2
sub2 = Flatten()(L3)
sub2 = Dense(528,activation='relu')(sub2) # 여기 조심 항상 전과 일치해야 한다.
sub2 = Dense(10,activation='relu')(sub2)

# layer_3 (continue)
L3 = Inception(L3,832)

# layer_4
L4 = Inception(L3,832)
L4 = Inception(L4,1024)
L4 = AveragePooling2D(pool_size=(4,4),padding='valid')(L4)
L4 = Dropout(0.4)(L4)
L4 = Flatten()(L4)
outputs = Dense(10,activation='softmax')(L4)

model = Model(inputs,[sub1,sub2,outputs]) # sub1,sub2,output이 softmax로 예측한 값이니 통합하여 이용 가능
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[0.3,0.3,1.0]) 

model.fit(x_train,y_train,batch_size=540, epochs=10)

score = model.evaluate(x_test,y_test)
print('loss: ', score[0])
print('acc: ', score[1])