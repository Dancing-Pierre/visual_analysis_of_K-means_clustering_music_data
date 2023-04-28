import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def pred(dataX):
    data = pd.read_csv("./Clustering/data.csv", encoding="utf-8")
    music = data['text']
    music = music.apply(remove)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(music)
    kmeans = KMeans(n_clusters=5, n_init='auto')
    kmeans.fit(X)
    names = data['title']
    label_map = {0: '经典老歌', 1: '流行', 2: '伤感情歌', 3: '网络热歌', 4: '民谣'}
    dataX = dataX.replace("\n", "")
    dataX = vectorizer.transform([dataX])
    #     return kmeans.predict(dataX)
    return label_map[kmeans.predict(dataX)[0]]


def remove(string):
    return string.replace("\n", "")

# X = '''我喜欢一回家　就有暖洋洋的灯光在等待
# 我喜欢一起床　就看到大家微笑的脸庞
# 我喜欢一出门　就为了家人和自己的理想打拼
# 我喜欢一家人　心朝着同一个方向眺望　哦～我喜欢快乐时　马上就想要和你一起分享
# 我喜欢受伤时　就想起你们温暖的怀抱
# 我喜欢生气时　就想到你们永远包容多么伟大
# 我喜欢旅行时　为你把美好记忆带回家因为我们是一家人　相亲相爱的一家人
# 有缘才能相聚　有心才会珍惜　何必让满天乌云遮住眼睛因为我们是一家人　相亲相爱的一家人
# 有福就该同享　有难必然同当　用相知相守还地久天长我喜欢一回家　就把乱遭遭的心情都忘掉
# 我喜欢一起床　就带给大家微笑的脸庞
# 我喜欢一出门　就为了个人和世界的美好打拼
# 我喜欢一家人　梦朝着同一个方向创造　哦～当别人快乐时　好像是自己获得幸福一样
# 当别人受伤时　我愿意敞开最真的怀抱
# 当别人生气时　告诉他就算观念不同不必激动
# 当别人需要时　我一定卷起袖子帮助他因为我们是一家人　相亲相爱的一家人
# 有缘才能相聚　有心才会珍惜　何必让满天乌云遮住眼睛因为我们是一家人　相亲相爱的一家人
# 有福就该同享　有难必然同当　用相知相守还地久天长※处处为你用心一直最有默契
# 　请你相信这份感情值得感激※因为我们是一家人　相亲相爱的一家人
# 有缘才能相聚　有心才会珍惜　何必让满天乌云遮住眼睛因为我们是一家人　相亲相爱的一家人
# 有福就该同享　有难必然同当　用相知相守还地久天长",相亲相爱一家人
# '''
# result = pred(X)
# print(result)
