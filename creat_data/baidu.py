from aip import AipNlp
from creat_data.config import account_baidu
import pandas as pd
import numpy as np
import json
import requests


# 逐句调用接口判断
def creat_label(texts, interface='SDK'):
    # 创建连接
    client = AipNlp(account_baidu['id_1']['APP_ID'],
                    account_baidu['id_1']['API_KEY'],
                    account_baidu['id_1']['SECRET_KEY'])
    results = []
    if interface == 'SDK':
        for one_text in texts:
            result = client.sentimentClassify(one_text)
            results.append([one_text,
                            result['items'][0]['sentiment'],
                            result['items'][0]['confidence'],
                            result['items'][0]['positive_prob'],
                            result['items'][0]['negative_prob']
                            ])
    elif interface == 'API':
        # 获取access_token
        url = 'https://aip.baidubce.com/oauth/2.0/token'
        params = {'grant_type': 'client_credentials',
                  'client_id': account_baidu['id_1']['API_KEY'],
                  'client_secret': account_baidu['id_1']['SECRET_KEY']}
        r = requests.post(url, params=params)
        access_token = json.loads(r.text)['access_token']

        url = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify'
        params = {'access_token': access_token}
        headers = {'Content-Type': 'application/json'}
        for one_text in texts:
            data = json.dumps({'text': one_text})
            r = requests.post(url=url,
                              params=params,
                              headers=headers,
                              data=data)
            result = json.loads(r.text)
            results.append([one_text,
                            result['items'][0]['sentiment'],
                            result['items'][0]['confidence'],
                            result['items'][0]['positive_prob'],
                            result['items'][0]['negative_prob']
                            ])

        # 逐句调用接口判断

        data = json.dumps({'text': '价格便宜啦，比原来优惠多了'})
        r = requests.post(url=url, params=params, headers=headers, data=data)
        result = json.loads(r.text)
    return results


if __name__ == '__main__':
    results = creat_label(texts=['价格便宜啦，比原来优惠多了',
                                 '壁挂效果差，果然一分价钱一分货',
                                 '东西一般般，诶呀',
                                 '快递非常快，电视很惊艳，非常喜欢',
                                 '到货很快，师傅很热情专业。'
                                 ])
    results = pd.DataFrame(results, columns=['evaluation',
                                             'label',
                                             'confidence',
                                             'positive_prob',
                                             'negative_prob'])
    results['label'] = np.where(results['label'] == 2, '正面', '负面')
    print(results)
