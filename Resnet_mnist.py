# 케라스
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Activation, add, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

width = 28
height = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, width, height, 1).astype('float32')/255.0
x_test = x_test.reshape(10000, width, height, 1).astype('float32')/255.0

num_classes = 10
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def Residual_Block(x, n_ch):
    skip_connection = x # 초기의 x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_ch, kernel_size=(3,3), strides=1, padding='same')(x) # 크기 안변하게 하는 패딩이 케라스에선 same
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_ch, kernel_size=(3,3), strides=1, padding='same')(x)
    x = add([x, skip_connection]) # 그냥 shape가 같은 배열을 요소별로 더하는 것이다.
    
    return x

inputs = Input(shape=(28,28,1))
x = Conv2D(64, kernel_size=7, strides=2, padding='valid')(inputs) # 크기가 변해야 하므로 (반으로 줄어야 하므로) 일단 valid
x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
x = Residual_Block(x, 64) # x = add([x, skip_connection])에 이미 x가 들어가 있기 때문에 x가 연결될 필요가 없다
x = Residual_Block(x, 64)
x = BatchNormalization()(x)
x = Conv2D(256, kernel_size=3, strides=2, padding='valid')(x)
x = Residual_Block(x, 256)
x = Residual_Block(x, 256)
x = GlobalAveragePooling2D()(x) # 빈 괄호는 option 없는 것

outputs = Dense(10, activation = 'softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=540, epochs=10)

score = model.evaluate(x_test,y_test)
print('loss: ', score[0])
print('acc: ', score[1])