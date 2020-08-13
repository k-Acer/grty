import os
import urllib.request
from pprint import pprint

import gensim
from gensim import corpora
import neologdn
import MeCab

def analyse_text(r_path, stop_words):
    """
    形態素解析を行う。

    Args:
        r_path (str): 読み込みファイル名
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析を行った単語をリスト型で返す
    """
    # テキストの読み込み
    with open(r_path) as f:
        text = f.read()
    # テキストの正規化
    text = neologdn.normalize(text)
    # 形態素解析
    m = MeCab.Tagger ("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    # # 確認用
    # print(m.parse(text))
    # 名詞の格納
    word_list = []
    node = m.parseToNode(text)
    while node:
        word = node.surface
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

def get_topick(texts):
    """
    トピックを抽出する。

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    # 辞書の作成
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    # 辞書をテキストファイルに保存
    # dictionary.save_as_text('./tmp/deerwester.dict.txt')
    corpus = [dictionary.doc2bow(text) for text in texts]
    # コーパスをファイルに保存
    # corpora.MmCorpus.serialize('./tmp/deerwester.mm', corpus)
    # num_topics=5で、5個のトピックを持つLDAモデルを作成
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
    # トピックの表示
    pprint(lda.show_topics())
        
def roop_analyse_text(path, stop_words):
    """
    analyse_text を繰り返し実行する。

    Args:
        path (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    # テキストのファイル名を取得
    files = os.listdir(path)
    # 各テキストに対して形態素解析
    texts = [analyse_text(path + f, stop_words) for f in files]
    return texts
    


if __name__ == '__main__':

    # ストップワードリストの作成
    stop_words = create_stopwords('./data/slothlib/slothlib.txt')

    # 形態素解析
    # gurunabi = analyse_text('./data/text/gurunabi.txt', stop_word)
    # hokuto = analyse_text('./data/text/hokuto.txt', stop_word)
    # idemitu = analyse_text('./data/text/idemitu.txt', stop_word)
    # komatuseisaku = analyse_text('./data/text/komatuseisaku.txt', stop_word)
    # koropura = analyse_text('./data/text/koropura.txt', stop_word)
    # morinaga = analyse_text('./data/text/morinaga.txt', stop_word)
    # roson = analyse_text('./data/text/roson.txt', stop_word)
    # takeda = analyse_text('./data/text/takeda.txt', stop_word)
    # teiseki = analyse_text('./data/text/teiseki.txt', stop_word)
    # toyota = analyse_text('./data/text/toyota.txt', stop_word)

    # # 確認用
    # print(roson)
    dir_path = './data/text_sample/'
    texts = roop_analyse_text(dir_path, stop_words)
    get_topick(texts)

    #### １年分のデータでやる

    
    