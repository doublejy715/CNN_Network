# import library
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D, BatchNormalization, ZeroPadding2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd

# solve memory error
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

# parameter
width = 28
height = 28

# read&set dataset
(train_x, train_y), (test_x,test_y) = mnist.load_data()
print(type(train_x))
print(train_x[0])
print(train_x.shape)
print(type(train_y))

train_x = train_x.reshape(60000, width, height, 1).astype('float32')/255.0
test_x = test_x.reshape(10000, width, height, 1).astype('float32')/255.0

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

#model(AlexNet)
Input_layer = Input(shape=[28,28,1])

#layer_1 (28x28x1)
H1_Conv = Conv2D(96,kernel_size=3,strides=1,activation='relu')(Input_layer) #이렇게 하면 kernel이 11,11,3 되나?
H1_Max = MaxPool2D(pool_size=(2,2),strides=2)(H1_Conv)
H1_Batch = BatchNormalization()(H1_Max)

#layer_2 (14x14x64)
H2_Zero = ZeroPadding2D(padding=(2,2))(H1_Batch) # 다른곳은 pad = 2 이렇게 해놨음
H2_Conv = Conv2D(256,kernel_size=3,strides=1,activation='relu')(H2_Zero)
H2_Max = MaxPool2D(pool_size=(2,2),strides=2)(H2_Conv)
H2_Batch = BatchNormalization()(H2_Max)

#layer_3 (8x8x256)
H3_Zero = ZeroPadding2D(padding=(1,1))(H2_Batch) 
H3_Conv = Conv2D(384,kernel_size=3,strides=1,activation='relu')(H3_Zero)
H3_Max = MaxPool2D(pool_size=(2,2),strides=2)(H3_Conv) # 이건 내가 1x1 맞추기 위해 넣음
H3_Batch = BatchNormalization()(H3_Max)

#layer_4 (4x4x384)
H4_Zero = ZeroPadding2D(padding=(1,1))(H3_Batch) 
H4_Conv = Conv2D(384,kernel_size=3,strides=1,activation='relu')(H4_Zero)
H4_Max = MaxPool2D(pool_size=(2,2),strides=2)(H4_Conv) # 이건 내가 1x1 맞추기 위해 넣음
H4_Batch = BatchNormalization()(H4_Max) # 이건 내가 1x1 맞추기 위해 넣음

#layer_5 (2x2x384)
H5_Zero = ZeroPadding2D(padding=(1,1))(H4_Batch) 
H5_Conv = Conv2D(256,kernel_size=3,strides=1,activation='relu')(H5_Zero)

#layer_6 flatten (1x1x256)
H6_flat = Flatten()(H5_Conv) #6x6x256 feature map -> 9216 size

#layer_7
H7_FC = Dense(4096,activation='relu')(H6_flat)
H7_Drop = Dropout(0.5)(H7_FC) # 이건 성능 상승위해 개별적으로 넣은것

#layer_8
H8_FC = Dense(1000,activation='relu')(H7_Drop)
H8_Drop = Dropout(0.5)(H8_FC)

#layer_9
Output_layer = Dense(10,activation='softmax')(H8_Drop)

# model_make
model = Model(Input_layer,Output_layer)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# practice
model.fit(train_x,train_y,batch_size=540,epochs=10)

# test code
score = model.evaluate(test_x,test_y) #[loss,acc]
print("loss : ",score[0])
print("acc : ",score[1])