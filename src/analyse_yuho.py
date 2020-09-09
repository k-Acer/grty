import os
import pickle
import re

import gensim
from gensim import corpora
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis.gensim
from tqdm import tqdm  # 進捗状況の表示
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

def make_lda_bow(texts, no_below, no_above, num_topics):
    """
    トピックを抽出する。
    Args:
        texts (list): 形態素解析を行った単語のリスト
        no_below (int): この数未満の文書にしか出現しない単語を切り捨てる
        no_above (float): 頻度が上位の単語を切り捨てる割合
        num_topics (int): トピック数

    Returns:
        corpus: コーパス
        lda: ldaモデル
        lda_param: ldaモデルのパラメーター
    """
    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # 辞書を保存
    # if not os.path.isdir('./tmp/dic'): os.mkdir('./tmp/dic')
    # if not os.path.isdir(dic_dir): os.mkdir(dic_dir)
    d_fname = './tmp/dic_{0}-{1}.txt'.format(no_below, int(no_above*100))
    if not os.path.isfile(d_fname): dictionary.save_as_text(d_fname)
    # コーパスの作成
    corpus = [dictionary.doc2bow(text) for text in texts]
    # コーパスをファイルに保存
    # if not os.path.isdir('./tmp/corpus'): os.mkdir('./tmp/corpus')
    # if not os.path.isdir(corpus_dir): os.mkdir(corpus_dir)
    # c_fname = os.path.join(corpus_dir, 'corpus_{0}-{1}.mm'.format(no_below, int(no_above*100)))
    # if not os.path.isfile(c_fname): corpora.MmCorpus.serialize(c_fname, corpus)
    # LDAモデルの作成
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, alpha=0.2, random_state=0)  # , alpha=0.2
    # ldaモデルの保存
    # if not os.path.isdir('./tmp/lda'): os.mkdir('./tmp/lda')
    # if not os.path.isdir(lda_dir): os.mkdir(lda_dir)
    # l_fname = os.path.join(lda_dir, 'lda_{0}-{1}-{2}.model'.format(no_below, int(no_above*100), num_topics))
    # if not os.path.isfile(l_fname): lda.save(l_fname)
    lda_param = [no_below, int(no_above*100), 'bow']
    return lda, lda_param, corpus, dictionary

# def make_hdp_bow(texts, no_below, no_above):
#     """
#     トピックを抽出する。
#     Args:
#         texts (list): 形態素解析を行った単語のリスト
#         no_below (int): この数未満の文書にしか出現しない単語を切り捨てる
#         no_above (float): 頻度が上位の単語を切り捨てる割合

#     Returns:
#         corpus: コーパス
#         lda: ldaモデル
#         lda_param: ldaモデルのパラメーター
#     """
#     print('Making LDA model by hdp ...')
#     # 辞書の作成
#     dictionary = corpora.Dictionary(texts)
#     dictionary.filter_extremes(no_below=no_below, no_above=no_above)
#     # コーパスの作成
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     # LDAモデルの作成
#     lda = gensim.models.hdpmodel.HdpModel(corpus=corpus, id2word=dictionary, random_state=0)
#     lda_param = [no_below, int(no_above*100), 'hdp_bow']
#     print('Fin')
#     return lda, lda_param, dictionary

# def make_lda_tfidf(texts, no_below, no_above, num_topics):
#     """
#     トピックを抽出する。
#     Args:
#         texts (list): 形態素解析を行った単語のリスト
#         no_below (int): この数未満の文書にしか出現しない単語を切り捨てる
#         no_above (float): 頻度が上位の単語を切り捨てる割合
#         num_topics (int): トピック数

#     Returns:
#         corpus: コーパス
#         lda: ldaモデル
#         lda_param: ldaモデルのパラメーター
#     """
#     # 辞書の作成
#     dictionary = corpora.Dictionary(texts)
#     dictionary.filter_extremes(no_below=no_below, no_above=no_above)
#     # 辞書を保存
#     d_fname = './tmp/dic_tfidf_{0}-{1}.txt'.format(no_below, int(no_above*100))
#     if not os.path.isfile(d_fname): dictionary.save_as_text(d_fname)
#     # コーパスの作成
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     tfidf = gensim.models.TfidfModel(corpus)
#     corpus_tfidf = tfidf[corpus]
#     # LDAモデルの作成
#     lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, random_state=0)
#     # ldaモデルの保存
#     lda_param = [no_below, int(no_above*100), 'tfidf']
#     return lda, lda_param, dictionary

def visualize_topics(lda, lda_param):
    """
    リスクトピックを表示する。

    Args:
        lda: ldaモデル
        lda_param (list): ldaモデルのパラメーター
    """
    ncols = 5
    nrows = 2
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

    # vis
    # plt.show()

    # ワードクラウドの保存
    fname = './tmp/{0}_{1}-{2}-{3}-2.jpg'.format(lda_param[2], lda_param[0], lda_param[1], lda.num_topics)
    if not os.path.isfile(fname): plt.savefig(fname)
    
def evaluate_model(w_file, texts, limit, b, a):
    """
    モデルの評価を行う。

    Args:
        w_file: 買い込みファイルのパス
        texts (list): 形態素解析を行った単語のリスト
        limit (int): 何トピックまで評価するか

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    #Metrics for Topic Models
    coherence_vals = []
    perplexity_vals = []

    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=b, no_above=a)
    # コーパスの作成
    corpus = [dictionary.doc2bow(text) for text in texts]
    # テスト用のコーパスを作成
    test_size = int(len(corpus) * 0.2)
    test_corpus = corpus[:test_size]
    # perplexity と coherence の取得
    for n_topic in tqdm(range(15, limit)):
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda.log_perplexity(test_corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())
    
    # evaluation
    x = range(15, limit)

    fig, ax1 = plt.subplots(figsize=(12,5))

    # coherence
    c1 = 'darkturquoise'
    ax1.plot(x, coherence_vals, 'o-', color=c1)
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

    # perplexity
    c2 = 'slategray'
    ax2 = ax1.twinx()
    ax2.plot(x, perplexity_vals, 'o-', color=c2)
    ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

    # Vis
    ax1.set_xticks(x)
    fig.tight_layout()
    # plt.show()

    # pngファイルを保存する
    plt.savefig(w_file)
    plt.close()

def get_topics(lda, dictionary, text_file):
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
    

def make_dataframe(lda, dictionary, target_dir):
    # コラムの作成
    columns = ['Topick_{0}'.format(i) for i in range(lda.num_topics)]
    columns.insert(0, '株式コード') # 株式コードの追加
    columns.append('Year') # 年度の追加
    # データの作成
    text_dir = os.path.join(target_dir, 'text') 
    # textディレクトリ下の各年度ディレクトリ名のリストを取得
    dirpaths = os.listdir(text_dir)
    # 全textファイルパスを取得
    filepaths = []
    for d in dirpaths:
        year_dir = os.path.join(text_dir, d)
        f = utils.get_filepaths(year_dir)
        for filepath in f:
            filepaths.append(filepath)
    # トピック分布のデータを作成
    data = []
    for filepath in filepaths:
        # トピック分布の取得
        topics_dist = get_topics(lda, dictionary, filepath)
        # 株式コードの追加
        basename = os.path.basename(filepath)  # ファイル名を取得
        name = os.path.splitext(basename)[0]  # 拡張子を削除
        topics_dist.insert(0, int(name))
        # 年度の追加
        year_dir = os.path.basename(os.path.dirname(filepath))
        topics_dist.append(int(year_dir))
        # データに加える
        data.append(topics_dist)
    # データフレーム化
    df = pd.DataFrame(data=np.array(data), columns=columns)
    return df


if __name__ == '__main__':

    # 全期間のトピックの取得
    texts = load_texts('./data/text_analysed/texts.pkl')
    # BoW
    lda, lda_param, corpus, dictionary = make_lda_bow(texts, 850, 0.2, 10)
    # hdp_boW
    # lda, lda_param, dictionary = make_hdp_bow(texts, 600, 0.2)
    # tfidf
    # lda, lda_param, dictionary = make_lda_tfidf(texts, 300, 0.2, 25)
    # トピックの表示
    # for t in lda.show_topics(-1, 20):
    #     print(t)

    visualize_topics(lda, lda_param)

    print(lda.alpha)

    # pyLDAVis
    # vis_pcoa = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    # pyLDAvis.save_html(vis_pcoa, 'pyldavis_pcoa_850_20_13_10.html')

    # データフレームの作成
    # target_dir = './data'
    # df = make_dataframe(lda, dictionary, target_dir)
    # CSVデータの作成
    # df.to_csv('risk_topics_575_20_25.csv')

    # モデルの評価
    # no_below = [700, 800, 900]
    # no_above = [0.1, 0.15]
    # for a in no_above:
    #     for b in no_below:
    #         above = int(a*100)
    #         w_path = './tmp/evaluation/all/evaluation_{0}_{1}.png'.format(b, above)
    #         evaluate_model(w_path, texts, 21, b, a)
    
    ### Sample
    # texts = load_texts('./data/sample/text_analysed/texts.pkl')
    # lda, lda_param, dictionary = make_lda_bow(texts, 1, 0.95, 5)
    # トピックの表示
    # for t in lda.show_topics(-1):
    #     print(t)
    # データフレームの作成
    # target_dir = './data/sample'
    # df = make_dataframe(lda, dictionary, target_dir)
    # CSVデータの作成
    # df.to_csv('sample.csv')


