import os
import re

from bs4 import BeautifulSoup
import chardet
import neologdn

import utils


def html2topics(read_file):
    """
    指定したhtmlファイルからテキストを抽出する。

    Args:
        read_file (str): 読み込みファイルのパス
    
    Returns:
        topic_list: テキストを特定の記号ごとに分割し、リストに格納して返す
    """
    # htmlファイルの読み込み
    try:
        with open(read_file, 'rb') as f:
            html = f.read()
        # 文字コードの判定とデコード
        if chardet.detect(html)['encoding'] == 'utf-8':
            html.decode('utf-8')
        else:
            html.decode('cp932')
    except UnicodeDecodeError:
        with open(read_file, 'r') as f:
            html = f.read()
    # htmlファイルからタグを取得
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find_all(['h3', 'h4', 'p', 'div'])
    # タグからテキストを取得し、重複を削除する
    uniq_texts = []
    for t in tag:
        if not t.text in uniq_texts:
            text = neologdn.normalize(t.text).strip()  # テキストの正規化 (stripは前後の空白を削除している)
            uniq_texts.append(text)
    # 取得したテキストをトピックに分割する
    topic_list = []
    topic = []
    meet_symbol = False
    symbol_is_paren = False
    symbol_is_round = False
    symbol_is_dot = False
    for text in uniq_texts:
        # 最初に取得した記号を基準にテキストをトピックに分割する
        if not meet_symbol:
            if re.match(r'[①-⑳]', text):
                symbol_is_round = True
                meet_symbol = True
            if re.match(r'\(\d+\)', text):
                symbol_is_paren = True
                meet_symbol = True
            if re.match(r'(\d+)\.\D', text):
                symbol_is_dot = True
                meet_symbol = True
        # トピックを取得
        if re.match(r'[①-⑳]', text) and symbol_is_round:
            if len(topic) > 0:
                topic_list.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        elif re.match(r'\(\d+\)', text) and symbol_is_paren:
            # print('Match')
            if len(topic) > 0:
                topic_list.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        elif re.match(r'(\d+)\.\D', text) and symbol_is_dot:
            if len(topic) > 0:
                topic_list.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        else:
            if len(topic) > 0:
                topic.append(text)
    if len(topic) > 0:
        topic_list.append(''.join(topic))
    return topic_list


def roop_html2topics(read_dir='./data'):
    """
    html2text をループする。
    topics と text を作成する。
    topics には企業の有報をトピックに分けて保存
    texts にはトピックを結合したものを保存

    Args:
        read_dir (str): 読み込みディレクトリのパス
    """
    # yuhoディレクトリ
    yuho_dir = os.path.join(read_dir, 'yuho')
    # トピック保存ディレクトリが存在しない場合は作成する。
    topics_dir = os.path.join(read_dir, 'topics') 
    if not os.path.isdir(topics_dir): os.mkdir(topics_dir)
    # テキスト保存ディレクトリが存在しない場合は作成する。
    texts_dir = os.path.join(read_dir, 'texts')
    if not os.path.isdir(texts_dir): os.mkdir(texts_dir)
    # yuhoディレクトリ下の各年度ディレクトリ名のリストを取得
    dirpaths = os.listdir(yuho_dir)
    # 全htmlファイルパスを取得
    filepaths = []
    for d in dirpaths:
        year_dir = os.path.join(yuho_dir, d)
        f = utils.get_filepaths(year_dir)
        for filepath in f:
            filepaths.append(filepath)
    # htmlファイルをテキストに変換し保存
    for f in filepaths:
        # htmlファイルからトピックを取得
        topics = html2topics(f)
        if len(topics) > 0:
            # 企業ディレクトリを作成
            basename = os.path.basename(f)  # 読み込みファイル名を取得
            name = os.path.splitext(basename)[0]  # htm拡張子を削除
            code = re.split('_', name)  # 株式コードを取得
            y_dir = os.path.dirname(f).replace('yuho', 'topics')  # topics下の年度ディレクトリの取得
            f_dir = os.path.join(y_dir, code[1])
            # topics下の年度ディレクトリがなければ作成
            if not os.path.isdir(y_dir): os.mkdir(y_dir)
            # topics下の企業ディレクトリがなければ作成
            if not os.path.isdir(f_dir): os.mkdir(f_dir)
            # ファイルが存在しなければ書き込み
            # for i, text in enumerate(topics):
                w_topic_file = os.path.join(f_dir, str(i) + '.txt')  # 書き込みファイル名
                try:
                    with open(w_topic_file, mode='x', encoding='utf-8') as f:
                        f.write(text)
                except FileExistsError:
                    pass
            # 企業テキストを保存
            y_dir_t = y_dir.replace('topics', 'texts')  # texts下の年度ディレクトリの取得
            w_text_file = os.path.join(y_dir_t, code[1] + '.txt')  # 書き込みファイル名
            # texts下の年度ディレクトリがなければ作成
            if not os.path.isdir(y_dir_t): os.mkdir(y_dir_t)
            firm_text = ''.join(topics)
            try:
                with open(w_text_file, mode='x', encoding='utf-8') as f:
                    f.write(firm_text)
            except FileExistsError:
                pass
                

if __name__ == '__main__':
    
    html2topic_all()

    ################ test
    ###### html2topics
    # read_file = './data/yuho/2006/10014_1377_yuho_101_20050531.htm'
    # topic_list = html2topics(read_file)
    # for topic in topic_list:
    #     print('---------------------------------')
    #     print(topic)
    ###### roop_html2text
    # read_dir = './data/sample'
    # roop_html2topics(read_dir)