import os
import pickle
import re
import urllib.request

import neologdn
import MeCab

import utils

def create_stopwords(path='./data/slothlib/slothlib.txt'):
    """
    ストップワードを作成する。

    Args:
        path (str): 読み込みファイル名

    Returns:
        list: ストップワードのリスト
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

def analyse_text(read_file, stop_words):
    """
    形態素解析を行う。

    Args:
        read_file (str): 読み込みファイルのパス
        stop_words (list): ストップワードのリスト

    Returns:
        list: 形態素解析を行った文書のリスト
    """
    # テキストの読み込み
    with open(read_file) as f:
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

def analyse_topics(read_dir, stop_words, start, end):
    """
    analyse_text を繰り返し実行する。

    Args:
        read_dir (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト
        start (int): この年から
        end (int): この年まで

    Returns:
        list: 形態素解析積みのテキストのリスト
    """
    # 年度ディレクトリ名のリストを取得
    dirpaths = os.listdir(read_dir)
    # 全期間のファイルパスを取得
    main_files = []
    for d in dirpaths:  # 年度ごと
        if (start <= int(d)) & (int(d) <= end):
            year_path = os.path.join(read_dir, d)
            # 企業ディレクトリ名のリストを取得
            firm_codes = os.listdir(year_path)
            for code in firm_codes:  # 企業ごと
                firm_path = os.path.join(year_path, code)
                file_paths = utils.get_filepaths(firm_path)
                for f in file_paths:
                    main_files.append(f)
    # 全期間のファイルに対して形態素解析
    print('Start Topics Analysis')
    text_list = [analyse_text(r_file, stop_words) for r_file in main_files]
    print('Fin Topics Analysis')
    return text_list

def analyse_texts(read_dir, stop_words, start, end):
    """
    analyse_text を繰り返し実行する。

    Args:
        read_dir (str): 読み込みディレクトリのパス
        stop_words (list): ストップワードのリスト
        start (int): この年から
        end (int): この年まで

    Returns:
        list: 形態素解析積みのテキストのリスト
    """
    # 年度ディレクトリ名のリストを取得
    dirpaths = os.listdir(read_dir)
    # 全期間のファイルパスを取得
    filepaths = []
    for d in dirpaths:
        if (start <= int(d)) & (int(d) <= end):
            year_path = os.path.join(read_dir, d)  # 年度パスの取得
            f = utils.get_filepaths(year_path)
            for filepath in f:
                filepaths.append(filepath)
    # 全期間のファイルに対して形態素解析
    print('Start Texts Analysis')
    texts = [analyse_text(r_file, stop_words) for r_file in filepaths]
    print('Fin Texts Analysis')
    return texts

def save_texts(write_file, text_list):
    '''
    形態素解析したテキストを pickle 形式で保存

    Args:
        write_file (str): 読み込みディレクトリのパス
        text_list (list): ストップワードのリスト
    '''
    # 書き込みディレクトリが存在しない場合は作成
    dirname = os.path.dirname(write_file)
    if not os.path.isdir(dirname): os.mkdir(dirname)
    # pickle 形式で保存
    with open(write_file, 'wb') as f:
        pickle.dump(text_list, f)



if __name__ == '__main__':

    # ストップワードリストの作成
    stop_words = create_stopwords('./data/slothlib/slothlib.txt')
    
    ########## 全期間の形態素解析
    ##### topics
    # read_topics_dir = './data/topics'
    # text_list = analyse_topics(read_topics_dir, stop_words, 2006, 2020)
    # write_topics_file = './data/analised_topics/topics.pkl'
    # save_texts(write_topics_file, text_list)

    ########## １年ごとの形態素解析
    ##### topics
    # year = [i for i in range(2011, 2021)]
    # for j in year:
    #     topics = analyse_topics(read_topics_dir, stop_words, j, j)
    #     w_file = './data/analised_topics/topics_{0}.pkl'.format(j)
    #     save_texts(w_file, topics)


    ################ sample
    # stop_words = create_stopwords('./data/slothlib/slothlib.txt')
    ##### topics
    # read_topics_dir = './data/sample/topics'
    # analised_topics = analyse_topics(read_topics_dir, stop_words, 2018, 2020)
    # write_topics_file = './data/sample/analised_topics/topics_2018-2020.pkl'
    # save_texts(write_topics_file, analised_topics)

    ###### pckle 確認用
    # r_file = './data/sample/analised_topics/topics_2018-2020.pkl'
    # with open(r_file, "rb") as f:
    #     hoge = pickle.load(f)
    # print(len(hoge))
