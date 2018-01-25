# Text-Classification

## 语言
Python3.5<br>
## 依赖库
pandas=0.21.0<br>
numpy=1.13.1<br>
scikit-learn=0.19.1<br>
gensim=3.2.0<br>
jieba=0.39<br>
keras=2.1.1<br>

## 方法介绍
项目介绍：通过对已有标签的帖子进行训练，实现新帖子的情感分类。现阶段通过第三方购买的数据，文本为爬虫抓取的电商购物评论，标签为“正面/负面”。

## 方法介绍
### 文本转tokenizer编码：sentence_2_tokenizer.py
先用jieba分词，再调用keras.preprocessing.text import Tokenizer转编码。<br>
``` python
# sentence_2_tokenizer(train_data,
#                      test_data=None,
#                      num_words=None,
#                      word_index=False)


# train_data: 训练集
# test_data: 测试集
# num_words: 词库大小,None则依据样本自动判定
# word_index: 是否需要索引
from sentence_transform.sentence_2_tokenizer import sentence_2_tokenizer

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
train_data_vec, test_data_vec, word_index = sentence_2_tokenizer(train_data=train_data,
                                                                 test_data=test_data,
                                                                 num_words=None,
                                                                 word_index=True)
```
![sentence_2_tokenizer](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_tokenizer.png)

### 文本转稀疏矩阵：sentence_2_sparse.py
先用jieba分词，再提供两种稀疏矩阵转换方式：1.转one-hot形式的矩阵，使用pandas的value_counts计数后转dataframe；2.sklearn.feature_extraction.text转成哈希表结构的矩阵。<br>
``` python
# sentence_2_sparse(train_data,
#                   test_data=None,
#                   language='Chinese',
#                   hash=True,
#                   hashmodel='CountVectorizer')

# train_data: 训练集
# test_data: 测试集
# language: 语种
# hash: 是否转哈希存储
# hashmodel: 哈希计数的方式
from sentence_transform.sentence_2_sparse import sentence_2_sparse

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
m, n = sentence_2_sparse(train_data=train_data, test_data=test_data, hash=True)
```
![ex1](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_sparse.png)

### 文本转词向量：sentence_2_vec.py
先用jieba分词，再调用gensim.models的word2vec计算词向量。<br>
``` python
# sentence_2_vec(train_data,
#                test_data=None,
#                size=5,
#                window=5,
#                min_count=1)

# train_data: 训练集
# test_data: 测试集
# size: 词向量维数
# window: word2vec滑窗大小
# min_count: word2vec滑窗内词语数量
from sentence_transform.sentence_2_vec import sentence_2_vec

train_data = ['全面从严治党',
              '国际公约和国际法',
              '中国航天科技集团有限公司']
test_data = ['全面从严测试']
train_data, test_data = sentence_2_vec(train_data=train_data,
                                       test_data=test_data,
                                       size=5,
                                       window=5,
                                       min_count=1)
```
![ex2](https://github.com/renjunxiang/Text-Classification/blob/master/picture/sentence_2_vec.png)

## 模型训练 models
### 监督机器学习：supervised_classify.py
利用sentence_transform.py文本转稀疏矩阵后，通过sklearn.feature_extraction.text模块转为哈希格式减小存储开销，然后通过常用的机器学习分类模型如SVM和KNN进行学习和预测。本质为将文本转为稀疏矩阵作为训练集的数据，结合标签进行监督学习。<br>
![ex3](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本分类.png)

### LSTM：LDA.py
keras的LSTM简单封装。<br>
``` python
# neural_LSTM(input_shape,
#             net_shape=[64, 64, 128, 2],
#             optimizer_name='Adagrad',
#             lr=0.001)

# input_shape: 样本数据格式
# net_shape: 神经网络格式
# optimizer_name: 优化器
# lr: 学习率
from models.neural_LSTM import neural_LSTM

model = neural_LSTM(input_shape=[10, 5],
                    net_shape=[64, 64, 128, 2],
                    optimizer_name='SGD',
                    lr=0.001)
model.summary()
```
![neural_LSTM](https://github.com/renjunxiang/Text-Classification/blob/master/picture/neural_LSTM.png)

### 非监督学习：LDA.py
``` python
# LDA(dataset=None,
#     topic_num=5,
#     alpha=0.0002,
#     beta=0.02,
#     steps=500,
#     error=0.1)

# dataset = 数据集,
# topic_num = 主题数,
# alpha = 学习率,
# beta = 正则系数,
# steps = 迭代上限,
# error = 误差阈值
from models.LDA import LDA

dataset = [['document' + str(i) for i in range(1, 11)],
           ['全面从严治党，是十九大报告的重要内容之一。十九大闭幕不久，习近平总书记在十九届中央纪委二次全会上发表重要讲话',
            '根据国际公约和国际法，对沉船进行打捞也要听取船东的意见。打捞工作也面临着很大的风险和困难，如残留凝析油可能再次燃爆',
            '下午召开的北京市第十四届人大常委会第四十四次会议决定任命殷勇为北京市副市长',
            '由中国航天科技集团有限公司所属中国运载火箭技术研究院抓总研制的长征十一号固体运载火箭“一箭六星”发射任务圆满成功',
            '直到2016年7月份，谢某以性格不合为由，向卢女士提出分手，并要求喝分手酒，可谁知，这醉翁之意不在酒哪',
            '湖北男子吴锐在其居住的湖南长沙犯下了一桩大案：跟踪一名开玛莎拉蒂女子',
            '甚而至于得罪了名人或名教授',
            '判决书显示，现年不到30岁的吴锐出生于湖北省天门市，住湖南省长沙县',
            '张某报警后，公安机关在侯某家门前将李某抢劫来的车辆前后别住。李某见状开始倒车',
            '被打女童来自哪里？打人者是谁？1月17日晚，澎湃新闻联系上女童曾某的母亲']]
model = LDA(dataset=dataset, steps=200)
document_topic, topic_word = model.document_recommend_topic(num_topic=2, num_word=8)
print('document_recommend_topic\n', document_topic)
print('topic_recommend_word\n', topic_word)
```
利用sentence_transform.py文本转稀疏矩阵后，对稀疏矩阵进行ALS分解，转为文本-主题矩阵*主题-词语矩阵。<br>
![ex4](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本主题分类数据.png)
![ex5](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本主题分类.png)

## DEMO
### 监督学习的范例：demo_score.py
读取数据集（商业数据暂时保密，仅提供部分预测结果约1400条），拆分数据为训练集和测试集，通过supervised_classify.py进行机器学习，再对每条文本打分。<br>
训练数据已更新,准确率最高84%<br>
![ex6](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_score_1.png)
图片为不同数据处理和不同模型的准确率<br>
![ex7](https://github.com/renjunxiang/Text-Classification/blob/master/picture/demo_score_2.png)

### 监督学习+打标签的范例：demo_topic_score.py
读取数据集NLP\data\，关键词：keyword.json，训练集train_data.json<br>，名称的配置文件config.py。然后通过supervised_classify.py对每个主题进行机器学习，再对每条文本打分。<br>
因为没有数据，我自己随便造了几句，训练效果马马虎虎~
![ex8](https://github.com/renjunxiang/Text-Classification/blob/master/picture/文本分类+打标签.png)




