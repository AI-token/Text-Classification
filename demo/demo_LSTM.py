from keras.models import Sequential
from keras.layers.core import Masking, Dense, initializers
from keras.layers.recurrent import LSTM
from keras.optimizers import Adagrad, SGD
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transform.sentence_2_sparse import sentence_2_sparse
from sentence_transform.sentence_2_vec import sentence_2_vec

positive = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='positive')
negative = pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                         sheet_name='negative')
# 分隔训练集和测试集
total = pd.concat([positive, negative], axis=0)
# 转词向量
data_transform = sentence_2_vec(train_data=total.loc[:, 'evaluation'],
                                test_data=None,
                                size=5,
                                window=5,
                                min_count=1)
# 将不同长度的文本进行'截断/填充'至相同长度,不设置maxlen则填充至最长
data_transform = pad_sequences(data_transform, maxlen=None, padding='post', value=0, dtype='float32')
label_transform = np.array(pd.get_dummies(total.loc[:, 'label']))
print(data_transform.shape)
# 拆分为训练集和测试集
train_data, test_data, train_label, test_label = train_test_split(data_transform,
                                                                  label_transform,
                                                                  test_size=0.33,
                                                                  random_state=42)

model = Sequential()
# 识别之前的'截断/填充',跳过填充
model.add(Masking(mask_value=0, input_shape=data_transform.shape[-2:]))
model.add(LSTM(units=64,
               activation='relu',
               implementation=1,
               dropout=0.2,
               kernel_initializer=initializers.normal(stddev=0.1),
               name='LSTM'))
# model.add(LSTM(units=64,
#                activation='relu',
#                # dropout=0.01,
#                implementation=1,
#                dropout=0.2,
#                name='LSTM'))
model.add(Dense(units=64,
                activation='relu',
                kernel_initializer=initializers.normal(stddev=0.1),
                name='Dense1'))
model.add(Dense(units=128,
                activation='relu',
                kernel_initializer=initializers.normal(stddev=0.1),
                name='Dense2'))
model.add(Dense(units=2,
                activation='softmax',
                kernel_initializer=initializers.normal(stddev=0.1),
                name='Dense3'))
adagrad = Adagrad(lr=0.0001)
# sgd=SGD(lr=0.0005)
model.compile(optimizer=adagrad, loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
model.fit(train_data, train_label, batch_size=50, epochs=20, verbose=1,
          validation_data=(test_data, test_label))
