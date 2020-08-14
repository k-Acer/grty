import os
import urllib.request
from pprint import pprint

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
        if hinshi == '名詞':
            word_list.append(word)
        node = node.next
    # 辞書によるストップワードの除去
    word_list = [word for word in word_list if word not in stop_words]
    return word_list

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

def make_lda(texts):
    """
    トピックを抽出する。
    Args:
        texts (list): 形態素解析を行った単語のリスト

    Returns:
        ldaモデル
    """
    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    # 辞書をテキストファイルに保存
    # dictionary.save_as_text('./tmp/deerwester.dict.txt')
    # コーパスの作成
    corpus = [dictionary.doc2bow(text) for text in texts]
    # コーパスをファイルに保存
    # corpora.MmCorpus.serialize('./tmp/deerwester.mm', corpus)
    # LDAモデルの作成
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=25, id2word=dictionary)
    # ldaモデルの保存
    # lda.save('./tmp/lda.model')
    return lda

def show_topics(lda):
    """
    リスクトピックを表示する。

    Args:
        ldaモデル
    """
    fig, axs = plt.subplots(ncols=5, nrows=int(lda.num_topics/5), figsize=(16,20))
    axs = axs.flatten()

    for i, t in enumerate(range(lda.num_topics)):
        x = dict(lda.show_topic(t, 30))
        im = WordCloud(font_path='./data/fonto/NotoSansCJK-Regular.ttc', background_color='white', width=300, height=300, random_state=0).generate_from_frequencies(x)
        axs[i].imshow(im)
        axs[i].axis('off')
        axs[i].set_title('Topic '+str(t))

    # vis
    # plt.tight_layout()
    # plt.show()

    # # ワードクラウドの保存
    plt.savefig('./tmp/wordcloud.png') 
        
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
    texts = [analyse_text(r_file, stop_words) for r_file in filepaths]
    return texts
    
def model_evaluation(w_path, texts):
    """
    トピック数ごとのモデルの評価を行う。  ## よく分かっていないので、修正が必要。

    Args:
        texts (list): 形態素解析を行った単語のリスト

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    #Metrics for Topic Models
    start = 2
    limit = 22
    step = 1
    coherence_vals = []
    perplexity_vals = []

    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    # コーパスの作成
    corpus = [dictionary.doc2bow(text) for text in texts]

    for n_topic in tqdm(range(start, limit, step)):
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())
    
    # evaluation
    x = range(start, limit, step)

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
    plt.show()

    # save as png
    plt.savefig(w_path) 


if __name__ == '__main__':

    # ストップワードリストの作成
    stop_words = create_stopwords('./data/slothlib/slothlib.txt')

    # 形態素解析
    # roson = analyse_text('./data/text_sample/roson.txt', stop_words)
    # 確認用
    # print(roson)

    # トピックの取得
    r_dir = './data/text/2019/'  # 2019年
    # r_dir = './data/text_sample/'  # sample
    texts = roop_analyse_text(r_dir, stop_words)
    lda = make_lda(texts)
    show_topics(lda)

    # # モデルの評価
    # w_path = './tmp/evaluation.png'
    # model_evaluation(w_path, texts)
    