import os
import pickle

import gensim
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

import morphological_analysis as ma
import utils

def load_texts(r_file):
    '''
    形態素解析したテキストを読み込む

    Args:
        r_file (str): 読み込みファイルのパス

    Returns:
        texts: 各文書の形態素解析結果のリスト
    '''
    if os.path.isfile(r_file):
        # pickle を読み込み
        print('Loading texts ...')
        with open(r_file, "rb") as f:
            texts = pickle.load(f)
        print('Fin')
    else:
        print('ファイルが存在しません')
    
    return texts

def sample(year, target_dir='./data'):
    """
    サンプル

    Args:
        year: 年度のリスト

    Returns:
        :
    """
    # 全年度のテキストを読み込む
    topics = load_texts('./data/analyzed_topics/topics.pkl')
    # 全年度のテキストから辞書を作成
    dictionary = corpora.Dictionary(topics)
    dictionary.filter_extremes(no_below=800, no_above=0.2)
    # 2006-2010 のテキストでコーパスを作成
    texts_2006_2010 = load_texts('./data/analyzed_topics/topics_2006_2010.pkl')
    corpus = [dictionary.doc2bow(text) for text in texts_2006_2010]
    # 学習
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=19, id2word=dictionary, alpha='auto', random_state=0)
    # 2006-2010 のトピックを可視化
    _visualize_topics(lda, 2010)
    # データフレーム用のリスト
    data = []
    # データの作成 2011-2020
    print('Start Making Data')
    for y in year: 
        print('--- Making {0} data ---'.format(y))
        # １年ごとのデータを加えていく
        add_texts = load_texts('./data/analyzed_topics/topics_{0}.pkl'.format(y))
        other_corpus= [dictionary.doc2bow(text) for text in add_texts]
        lda.update(other_corpus)
        # データの可視化
        _visualize_topics(lda, y)
        # データの作成
        data = _make_data(lda, dictionary, target_dir, data, y)
    print('Fin Making Date')
    # コラムの作成
    columns = ['Topick_{0}'.format(i) for i in range(lda.num_topics)]  # トピック
    columns.insert(0, '株式コード') # 株式コードの追加
    columns.append('Year') # 年度の追加
    # データフレームの作成
    df = pd.DataFrame(data=np.array(data), columns=columns)
    # csv で保存
    df.to_csv('risk_topics-bow_800_20_19_auto.csv')

### make_data で加えるデータは１年分
def _make_data(lda, dictionary, target_dir, data, year):
    # データの作成
    text_dir = os.path.join(target_dir, 'texts') 
    # 指定年度のtextファイルパスを取得
    year_dir = os.path.join(text_dir, str(year))
    filepaths = utils.get_filepaths(year_dir)
    # トピック分布のデータを作成
    for filepath in filepaths:
        # トピック分布の取得
        topics_dist = _get_topics(lda, dictionary, filepath)
        # 株式コードの追加
        basename = os.path.basename(filepath)  # ファイル名を取得
        name = os.path.splitext(basename)[0]  # 拡張子を削除
        topics_dist.insert(0, name)
        # 年度の追加
        topics_dist.append(str(year))
        # データに加える
        data.append(topics_dist)
    return data

def _get_topics(lda, dictionary, text_file):
    """
    各有価証券報告書のトピック分布を推定する。

    Args:
        lda_model: LDAモデルのパス

    Returns:
        list: 各有価証券報告書のトピック分布
    """
    # ターゲット文書の形態素解析
    stop_words = ma.create_stopwords('./data/slothlib/slothlib.txt')
    text = ma.analyse_text(text_file, stop_words)
    # ベクトル化
    vec = dictionary.doc2bow(text)
    # 分類結果表示
    topics_dist = [0] * lda.num_topics
    for i in lda[vec]:  # lda.get_document_topics(vec) でも同じような結果が出るが
        topics_dist[i[0]] = i[1]
    return topics_dist

def _visualize_topics(lda, name):
    """
    リスクトピックを表示する。

    Args:
        lda: ldaモデル
        name: ファイル名
    """
    ncols = 5
    nrows = 4
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,16))
    axes = axes.flatten()

    for i in range(lda.num_topics):
        x = dict(lda.show_topic(i, 30))
        im = WordCloud(font_path='./data/fonto/NotoSansCJK-Regular.ttc', background_color='white', width=300, height=300, random_state=0).generate_from_frequencies(x)
        axes[i].imshow(im)
        axes[i].axis('off')
        axes[i].set_title('Topic '+ str(i))

    for i in range(lda.num_topics, ncols*nrows):
        axes[i].axis('off')

    # ワードクラウドの保存
    fname = './tmp/{0}.jpg'.format(name)
    if not os.path.isfile(fname): plt.savefig(fname)



if __name__ == '__main__':
    year = [i for i in range(2011, 2021)]
    target_dir = './data'
    sample(year, target_dir)
