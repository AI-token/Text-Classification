import numpy as np
import pandas as pd
from supervised_classify import supervised_classify
from sklearn.model_selection import train_test_split

#读取正负面标签数据
positive=pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                   sheet_name='positive')
negative=pd.read_excel('D:/github/Text-Classification/data/demo_score/data.xlsx',
                   sheet_name='negative')

test_data=pd.read_excel('D:/github/Text-Classification/test/6000条原帖.20180125.xlsx',
                   sheet_name='Sheet1')
#分隔训练集和测试集
total=pd.concat([positive,negative],axis=0)
X_train, X_test, y_train, y_test = train_test_split(total.loc[:, 'evaluation'],
                                                    total.loc[:, 'label'],
                                                    test_size=0.33,
                                                    random_state=42)

result = supervised_classify(language='Chinese',
                             model_exist=False,
                             model_path=None,
                             model_name='SVM',
                             hashmodel=None,
                             vector=True,
                             savemodel=False,
                             train_dataset=[list(X_train), list(y_train)],
                             test_data=list(test_data.iloc[:,0]))
predict_evaluate = pd.DataFrame({'document': test_data.iloc[:,0],
                                 'predict': result},
                                columns=['document',  'predict'])
predict_evaluate=predict_evaluate.reset_index(drop=True)
predict_evaluate.to_excel('D:/github/Text-Classification/test/predict.xlsx',
                                index=False)#分类错误的数据保存下来
