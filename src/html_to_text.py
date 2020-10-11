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
    tag = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'span'])
    # タグからテキストを取得し、重複を削除する
    uniq_texts = []
    for t in tag:
        text = neologdn.normalize(t.text).strip()  # テキストの正規化 (stripは前後の空白を削除している)
        if not text in uniq_texts:
            uniq_texts.append(text)
    # 取得したテキストをトピックに分割する
    topic_list = []
    topic = []
    symbol = None
    symbol_list = []
    header_num = 0
    for text in uniq_texts:
        # 最初に取得した記号を基準にテキストをトピックに分割する
        if not symbol and not re.search(r'事業等のリスク', text):
            if re.match(r'①', text):  # 丸付き数字
                symbol = r'[①-⑳]'
            elif re.match(r'\(1\)', text):  # 括弧+数字
                symbol = r'\(\d+\)'
            elif re.match(r'\([aA]\)', text):  # 括弧+アルファベッ
                symbol = r'\([a-zA-Z]\)'
            elif re.match(r'\([ア-ン]\)', text):  # 括弧+カナ
                symbol = r'\([ア-ン]\)'
            elif re.match(r'1\.(\D)(\D)+', text):  # 数字.
                symbol = r'(\d+)\.(\D)(\D)+'
            elif re.match(r'[aA]\.(\D)(\D)+', text):  # アルファベット.
                symbol = r'[a-zA-Z]\.(\D)(\D)+'
            elif re.match(r'[ア-ン]\.(\D)(\D)+', text):  # カナ.
                symbol = r'[ア-ン]\.(\D)(\D)+'
            elif re.match(r'1\)', text):  # 半括弧+数字
                symbol = r'\d+\)'
            elif re.match(r'[aA]\)', text):  # 半括弧+アルファベッ
                symbol = r'[a-zA-Z]\)'
            elif re.match(r'[ア-ン]\)', text):  # 半括弧+カナ
                symbol = r'[ア-ン]\)'
            elif re.match(r'1(\D)(\D)+', text):  # 数字
                symbol = r'(\d)(\D)(\D)+'
            elif re.match(r'�@', text):  # 文字化け
                symbol = r'�[A-Z@]'
            else:
                header_num+=1
                if header_num > 20:
                    print('Too many header')
                    break
        # トピックを取得
        if symbol and re.match(symbol, text):
            this_symbol = re.match(symbol, text).group()
            if (not this_symbol in symbol_list) and len(symbol_list) > 0:  # シンボルが同じでないならば
                topic_list.append(''.join(topic))
                topic.clear()
                topic.append('{0}。'.format(text))
                symbol_list.append(this_symbol)
            elif len(symbol_list) == 0:  # はじめにマッチしたときのみここを通る
                topic.append('{0}。'.format(text))
                symbol_list.append(this_symbol)
        else:
            if len(topic) > 0:
                topic.append(text)
    # topic_list が空の場合
    if len(topic_list) == 0:
        header_num = 0
        for text in uniq_texts:
            if not symbol and not re.search(r'事業等のリスク', text):
                if re.match(r'\([ぁ-んァ-ン一-龥・]*\)', text):  # 括弧 + 文字
                    print('MATCH')
                    symbol = r'\([ぁ-んァ-ン一-龥・]*\)'
                elif re.match(r'〔[ぁ-んァ-ン一-龥・]*〕', text):  # 鈎括弧 + 文字
                    print('MATCH')
                    symbol = r'〔[ぁ-んァ-ン一-龥・]*〕'
                else:
                    header_num+=1
                    if header_num > 20:
                        print('Also too many header')
                        break
            # トピックを取得
            if symbol and re.match(symbol, text):
                this_symbol = re.match(symbol, text).group()
                if (not this_symbol in symbol_list) and len(symbol_list) > 0:  # シンボルが同じでないならば
                    topic_list.append(''.join(topic))
                    topic.clear()
                    topic.append('{0}。'.format(text))
                    symbol_list.append(this_symbol)
                elif len(symbol_list) == 0:  # はじめにマッチしたときのみここを通る
                    topic.append('{0}。'.format(text))
                    symbol_list.append(this_symbol)
            else:
                if len(topic) > 0:
                    topic.append(text)
    # 最後のトピックを加える
    if len(topic) > 0:
        topic_list.append(''.join(topic))
    # print('{0} has {1} header'.format(read_file, header_num))
    return topic_list


def roop_html2topics(read_dir='./data'):
    """
    html2text をループする。
    topics と text を作成する。
    topics には企業の有報をトピックに分けて保存
    texts にはトピックを結合したテキストを保存

    Args:
        read_dir (str): 読み込みディレクトリのパス
    """
    # yuhoディレクトリ
    yuho_dir = os.path.join(read_dir, 'yuho')
    # トピック保存ディレクトリが存在しない場合は作成する。
    topics_dir = os.path.join(read_dir, 'topics') 
    if not os.path.isdir(topics_dir): os.mkdir(topics_dir)
    # 企業テキスト保存ディレクトリが存在しない場合は作成する。
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
    only_one_topic = 0
    for f in filepaths:
        # htmlファイルからトピックを取得
        topics = html2topics(f)
        if len(topics) > 1:  # トピックが１しかない場合、うまく抽出できていない可能性が高い
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
            for i, text in enumerate(topics):
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
        else:
            print('{0} have less than one topic: {1}'.format(f, len(topics)))
            only_one_topic+=1
    print('There are {0} firms that have less than one topic'.format(only_one_topic))
                

if __name__ == '__main__':
    
    roop_html2topics()

    ################ test
    ###### html2topics
    # read_file = './data/yuho/2020/10014_8135_yuho_101_20200331.htm'
    # topic_list = html2topics(read_file)
    # print('SHOW TOPICS')
    # for topic in topic_list:
    #     print('---------------------------------')
    #     print(topic)

    ###### roop_html2text
    # roop_html2topics('./data/sample')