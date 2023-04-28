# -*- coding:utf-8 -*-
import requests
import matplotlib.pyplot as plt
import json
import re
import os
import jieba
import wordcloud
from wordcloud import WordCloud
import PIL.Image as image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import pandas as pd
import jieba.analyse
from snownlp import SnowNLP
import requests
from bs4 import BeautifulSoup

from Clustering.clustering import pred


def write_into_file(text1):
    path = os.getcwd() + '/'
    files = os.listdir(path)
    # 查找文件
    for f in files:
        if f == 'music_words.txt':
            print("有该歌词文件")
            with open(f, 'w', encoding='utf-8') as file:
                file.write(text1)
                print("歌词写入完成")
                file.close()
                break
    else:
        print("没有该歌词文件")

        def create_text(filename):
            path = os.getcwd() + '/'  # 当前目录为文件路径
            file_path = path + filename + '.txt'
            print("已经帮你自动创建歌词文件")
            file = open(file_path, 'w')
            try:
                file.write(text1)
                print("歌词写入完成")
            except:
                return ("歌词写入异常")
            file.close()

        create_text('music_words')  # 调用函数


def craw(list_ele, print_io):
    headers = {
        "User-Agent": "Mozilla/5.0(Windows NT 10.0; WOW64) AppleWebKit/537.36(KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36 ",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Accept-Charset": "GB2312,utf-8;q=0.7,*;q=0.7"
    }
    url = f'https://music.163.com/playlist?id={str(list_ele)}'
    s = requests.session()
    s = BeautifulSoup(s.get(url, headers=headers).content, features='lxml')
    main = s.find('ul', {'class': 'f-hide'})

    music_list_content = []
    data_list = []
    jindian = ''
    liuxing = ''
    qingge = ''
    wangluo = ''
    mingyao = ''

    for music in main.find_all('a'):
        title = music.text
        music_id = int(music['href'].split('=')[1])

        url = 'http://music.163.com/api/song/lyric?' + \
              'id=' + str(music_id) + '&lv=1&kv=1&tv=-1'

        # 用这行代码可以绕过网易云的token请求
        r = requests.get(url, headers=headers, allow_redirects=False)
        # allow_redirects设置为重定向的参数
        # headers=headers添加请求头的参数，冒充请求头

        # 用js将获取的歌词源码进行解析
        json_obj = r.text  # .text返回的是unicode 型的数据，需要解析
        j = json.loads(json_obj)  # 进行json解析
        words = j['lrc']['lyric']  # 将解析后的歌词存在words变量中

        # 解析后的歌词发现每行歌词前面有时间节点，将它进行美化一下：
        pattern = '\\(.*?\\)|\\{.*?}|\\[.*?]'
        text1 = re.sub(pattern, "", words)  # 用正则表达式将时间剔除
        text1 = text1.replace('编曲', '').replace('弦乐', '').replace('作词', '').replace('作曲', '').replace('录音棚',
                                                                                                              '').replace(
            '录音室', '').replace('母带', '').replace('电吉他', '').replace('录音师', '').replace('混音', '')
        # 通过歌词进行聚类，分析是什么类别的音乐
        result = pred(text1)
        sentiment = sentiment_analysis(text1, title)
        print(title)
        last_result = '{} 音乐类型是：{}，歌曲的情感类型是：{}'.format(title, result, sentiment)
        # 经典老歌、流行、伤感情歌、网络热歌、民谣
        if result == '经典老歌':
            jindian = jindian + text1
        elif result == '流行':
            liuxing = liuxing + text1
        elif result == '伤感情歌':
            qingge = qingge + text1
        elif result == '网络热歌':
            wangluo = wangluo + text1
        elif result == '民谣':
            mingyao = mingyao + text1
        data_list.append(last_result)
        music_list_content.append(text1)
        print('===========================================================')
    # 生成词云
    creatcloud(jindian, '经典老歌')
    creatcloud(liuxing, '流行')
    creatcloud(qingge, '伤感情歌')
    creatcloud(wangluo, '网络热歌')
    creatcloud(mingyao, '民谣')
    print(data_list)
    return music_list_content


# 词云
def creatcloud(text, title):
    if text:
        ls = jieba.lcut(text)  # 生成分词列表
        text = ' '.join(ls)  # 连接成字符串
        wc = wordcloud.WordCloud(font_path="msyh.ttc",
                                 width=1000,
                                 height=700,
                                 background_color='white',
                                 max_words=100)
        # msyh.ttc电脑本地字体，写可以写成绝对路径
        wc.generate(text)  # 加载词云文本
        wc.to_file("{}.png".format(title))  # 保存词云文件
        print('{}词云图已生成'.format(title))
    else:
        print('该歌单没有{}类的歌，故无法生成词云'.format(title))


# 输入歌单id
def inputer():
    val = int(input("请输入歌单列表id: "))
    return val


# 情感分析
def sentiment_analysis(text, title):
    # 对歌曲类别进行 K 均值聚类分析
    song_info = "\n".join(
        [line for line in text.split("\n") if ":" or " : " not in line])

    keywords = jieba.analyse.extract_tags(
        song_info, topK=20, withWeight=True, allowPOS=('n', 'v', 'a'))  # 抽取关键词

    df = pd.DataFrame(keywords, columns=['word', 'weight'])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(df[['weight']])  # 聚类算法
    df['class'] = kmeans.labels_  # 将聚类结果添加到 DataFrame 中
    print(df)
    # 分词
    words = jieba.lcut(song_info)

    # 情感词典
    with open('./Txt/right part.txt', 'r', encoding='gbk') as f:
        positive_words = [line.strip() for line in f.readlines()]
        f.close()
    with open('./Txt/error part.txt', 'r', encoding='gbk') as f:
        negative_words = [line.strip() for line in f.readlines()]
        f.close()

    # 统计情感词汇数量
    positive_count = 0
    negative_count = 0
    for word in words:
        if word in positive_words:
            positive_count += 1
        elif word in negative_words:
            negative_count += 1
    sentiment = ''
    # 计算情感极性
    if positive_count > negative_count:
        sentiment = '积极'
    elif positive_count < negative_count:
        sentiment = '消极'

    # 生成柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.bar(df.index, df['weight'], color=df['class'].map(
        {0: 'red', 1: 'green', 2: 'blue'}))
    plt.xticks(df.index, df['word'], rotation=90)
    plt.title('{}歌曲类别分析结果'.format(title))
    plt.xlabel('Keyword')
    plt.ylabel('Weight')
    plt.show()

    return sentiment


if __name__ == '__main__':
    playlist_id = inputer()
    # 爬取内容
    inner = craw(playlist_id, 0)
    text1 = ''.join(inner)
    # 写入文件
    write_into_file(text1)
