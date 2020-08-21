import os
from pprint import pprint
import re
import urllib.request

import gensim
from gensim import corpora
import neologdn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import MeCab
from tqdm import tqdm  # 進捗状況の表示
from wordcloud import WordCloud

import utils

def create_stopwords(path):
    """
    ストップワードを作成する。

    Args:
        path (str): 読み込みファイル名

    Returns:
        list: ストップワードをリスト型で返す
    """
    # パスが存在しない場合、slothlib をダウンロード
    if not os.path.exists(path):
        print('slothlib をダウンロード')
        download_slothlib()
    # ファイルの読み込み
    with open(path) as f:
        stop_words = f.read().splitlines()
    # 空文字列の削除
    stop_words = [word for word in stop_words if word != '']
    return stop_words

def download_slothlib(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    urllib.request.urlretrieve(url, path)

def analyse_text(r_file, stop_words):
    """
    形態素解析を行う。

    Args:
        r_file (str): 読み込みファイルのパス
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析を行った単語をリスト型で返す
    """
    # テキストの読み込み
    with open(r_file) as f:
        text = f.read()
    # テキストの正規化
    text = neologdn.normalize(text)
    # 形態素解析
    m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    # 名詞の格納
    word_list = []
    node = m.parseToNode(text)
    while node:
        word = node.feature.split(",")[6]
        hinshi = node.feature.split(",")[0]
        hinshi_shurui = node.feature.split(",")[1]
        if hinshi == '名詞' and (hinshi_shurui == '一般' or hinshi_shurui == '固有名詞' or hinshi_shurui == 'サ変接続'):
            if word == '*':
                word_list.append(node.surface)
            else:
                word_list.append(word)
        node = node.next
    # 辞書によるストップワードの除去
    word_list = [word for word in word_list if word not in stop_words]
    # 正規表現によるストップワードの削除
    word_list = [normalize_texts(word) for word in word_list]
    word_list = [word for word in word_list if word != '']
    return word_list

def normalize_texts(word):
    word = re.sub('\d+年', '', word)
    word = re.sub('\d+年度', '', word)
    word = re.sub('\d+月', '', word)
    word = re.sub('\d+日', '', word)
    word = re.sub('\d+月\d+日', '', word)
    word = re.sub('\d+億円?', '', word)
    word = re.sub('.*%', '', word)
    return word

def roop_analyse_text(r_dir, stop_words):
    """
    analyse_text を繰り返し実行する。

    Args:
        r_dir (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    # テキストのファイル名を取得
    filepaths = utils.get_filepaths(r_dir)
    # 各テキストに対して形態素解析
    print('Analysing Text ...')
    texts = [analyse_text(r_file, stop_words) for r_file in filepaths]
    print('Fin')
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
        lda: ldaモデル
        lda_param: ldaモデルのパラメーター
    """
    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # 辞書をテキストファイルに保存
    d_fname = './tmp/dic_{0}-{1}.txt'.format(no_below, int(no_above*10))
    if not os.path.isfile(d_fname): dictionary.save_as_text(d_fname)
    # コーパスの作成
    corpus = [dictionary.doc2bow(text) for text in texts]
    # コーパスをファイルに保存
    # corpora.MmCorpus.serialize('./tmp/deerwester.mm', corpus)
    # LDAモデルの作成
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=0)
    # ldaモデルの保存
    # lda.save('./tmp/lda_bow.model')
    lda_param = [no_below, int(no_above*10), num_topics]
    return lda, lda_param

########### BoW が片付いたら開発をすすめる。
# def make_lda_tfidf(texts):
#     """
#     トピックを抽出する。
#     Args:
#         texts (list): 形態素解析を行った単語のリスト

#     Returns:
#         ldaモデル
#     """
#     # 辞書の作成
#     dictionary = corpora.Dictionary(texts)
#     dictionary.filter_extremes(no_below=2, no_above=0.8)
#     # コーパスの作成
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     # LDAモデルの作成
#     tfidf = gensim.models.TfidfModel(corpus)
#     corpus_tfidf = tfidf[corpus]
#     lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, num_topics=25, id2word=dictionary)
#     # ldaモデルの保存
#     # lda.save('./tmp/lda_tfidf.model')
#     return lda

def visualize_topics(lda, lda_param):
    """
    リスクトピックを表示する。

    Args:
        lda: ldaモデル
        lda_param (list): ldaモデルのパラメーター
    """
    fig, axs = plt.subplots(ncols=5, nrows=int(lda.num_topics/5), figsize=(16,20))
    axs = axs.flatten()

    for i, t in enumerate(range(lda.num_topics)):
        x = dict(lda.show_topic(t, 30))
        im = WordCloud(font_path='./data/fonto/NotoSansCJK-Regular.ttc', background_color='white', width=300, height=300, random_state=0).generate_from_frequencies(x)
        axs[i].imshow(im)  # index 10 is out of bounds for axis 0 with size 10
        axs[i].axis('off')
        axs[i].set_title('Topic '+str(t))

    # vis
    # plt.tight_layout()
    # plt.show()

    # ワードクラウドの保存
    fname = './tmp/{0}-{1}_{2}.jpg'.format(lda_param[0], lda_param[1], lda_param[2])
    if not os.path.isfile(fname): plt.savefig(fname)
    
def model_evaluation(w_file, texts, limit, b, a):
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
    for n_topic in tqdm(range(2, limit)):
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda.log_perplexity(test_corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())
    
    # evaluation
    x = range(2, limit)

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


if __name__ == '__main__':

    # ストップワードリストの作成
    stop_words = create_stopwords('./data/slothlib/slothlib.txt')

    # # 形態素解析
    # text = analyse_text('./data/text_sample/roson.txt', stop_words)
    # # 確認用
    # print(text)

    # トピックの取得
    r_dir = './data/text/2019/'  # 2019年
    texts = roop_analyse_text(r_dir, stop_words)
    # # BoW
    lda, lda_param = make_lda_bow(texts, 80, 0.7, 10)
    # # TFIDF
    # # lda = make_lda_tfidf(texts)

    # # トピックの表示
    # # for t in lda.show_topics(-1):
    # #     print(t)

    visualize_topics(lda, lda_param)

    # モデルの評価
    # no_below = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # no_above = [0.4, 0.5, 0.6, 0.7, 0.8]
    # for a in no_above:
    #     for b in no_below:
    #         above = int(a*10)
    #         w_path = './tmp/evaluation_{0}_{1}.png'.format(b, above)
    #         model_evaluation(w_path, texts, 30, b, a)

    

    # [Test] トピックの取得
    # r_dir = './data/text_sample/'
    # texts = roop_analyse_text(r_dir, stop_words)
    # lda = make_lda_bow(texts, './tmp/sample_re_2019_1_9_10.txt')
    # for t in lda.show_topics(-1):
    #     print(t)


    ################  精度を上げる！！！！！！！！！！！！！！！！
    #### 固有名詞ありよりも、一般名詞のみのほうが良いかも
    ## まずは　BoW

    # 100-8-8 0.29-110, 90-8-12 0.29弱-120
    