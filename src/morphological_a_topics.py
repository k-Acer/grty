import os
import pickle
import re
import urllib.request

import neologdn
import MeCab

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
    word = re.sub(r'\d+年', '', word)
    word = re.sub(r'\d+年度', '', word)
    word = re.sub(r'\d+月', '', word)
    word = re.sub(r'\d+日', '', word)
    word = re.sub(r'\d+月\d+日', '', word)
    word = re.sub(r'\d+億円?', '', word)
    word = re.sub(r'.*%', '', word)
    return word

def analyse_text_roop(r_dir, stop_words, start, end):
    """
    analyse_text を繰り返し実行する。

    Args:
        r_dir (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト
        start (int): この年から
        end (int): この年まで

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    """
    texts = ['0']
    # テキストファイルが格納されている各年度のディレクトリ名のリストを取得
    dirpaths = os.listdir(r_dir)
    # 全期間のファイルパスを取得
    main_files = []
    for d in dirpaths:  # 年度
        if (start <= int(d)) & (int(d) <= end):
            year_path = os.path.join(r_dir, d)
            firm_codes = os.listdir(year_path)
            for code in firm_codes:  # 企業
                firm_path = os.path.join(year_path, code)
                file_paths = utils.get_filepaths(firm_path)
                for f in file_paths:
                    main_files.append(f)
    # 全期間のファイルに対して形態素解析
    print('Start Text Analysis')
    texts = [analyse_text(r_file, stop_words) for r_file in main_files]
    print('Fin Text Analysis')
    return texts

def save_texts(w_file, texts):
    '''
    形態素解析したテキストを pickle 形式で保存

    Args:
        r_dir (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析積みの各テキストをリストに格納して返す
    '''
    # 書き込みディレクトリが存在しない場合は作成
    dirname = os.path.dirname(w_file)
    if not os.path.isdir(dirname): os.mkdir(dirname)
    # pickle 形式で保存
    with open(w_file, 'wb') as f:
        pickle.dump(texts, f)
    



if __name__ == '__main__':

    # ストップワードリストの作成
    stop_words = create_stopwords('./data/slothlib/slothlib.txt')
    r_dir = './data/topics'
    texts = analyse_text_roop(r_dir, stop_words, 2006, 2020)
    w_file = './data/topics_analysed/topics.pkl'
    save_texts(w_file, texts)

    ##### 特定期間の形態素解析
    # stop_words = create_stopwords('./data/slothlib/slothlib.txt')
    # r_dir = './data/text'
    # year = [i for i in range(2011, 2021)]
    # for j in year:
    #     texts = analyse_text_roop(r_dir, stop_words, j, j)
    #     w_file = './data/text_analysed/texts_{0}.pkl'.format(j)
    #     save_texts(w_file, texts)

    ################ sample
    # stop_words = create_stopwords('./data/slothlib/slothlib.txt')
    # r_dir = './data/sample/topics'
    # texts = analyse_text_roop(r_dir, stop_words, 2018, 2020)
    # w_file = './data/sample/topics_analysed/topics_2020.pkl'
    # save_texts(w_file, texts)

    ###### pckle 確認用
    # with open(w_file, "rb") as f:
    #     hoge = pickle.load(f)
    # print(len(hoge))