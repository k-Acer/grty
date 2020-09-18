import os
import re

from bs4 import BeautifulSoup
import chardet
import neologdn

import utils


def html2topic(r_file):
    # テキストをバイト列として読み込み
    with open(r_file, 'rb') as f:
        html = f.read()
    # 文字コードの判定とデコード
    if chardet.detect(html)['encoding'] == 'utf-8':
        html.decode('utf-8')
    else:
        html.decode('cp932')
    # テキストの取得
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find_all(['h3', 'h4', 'p'])
    # 重複を削除する
    uniq_texts = []
    for t in tag:
        if not t.text in uniq_texts:
            text = neologdn.normalize(t.text).strip()  # テキストの正規化 (stripは前後の空白を削除している)
            uniq_texts.append(text)
    # テキストをトピックに結合する、
    topics = []
    topic = []
    meet_symbol = False
    symbol_is_paren = False
    symbol_is_round = False
    symbol_is_dot = False
    for text in uniq_texts:
        # print('--------------------------------------')
        # print(text)
        # 最初に取得した記号をトピックの基準にする
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
        # トピックを結合
        if re.match(r'[①-⑳]', text) and symbol_is_round:
            if len(topic) > 0:
                topics.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        elif re.match(r'\(\d+\)', text) and symbol_is_paren:
            # print('Match')
            if len(topic) > 0:
                topics.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        elif re.match(r'(\d+)\.\D', text) and symbol_is_dot:
            if len(topic) > 0:
                topics.append(''.join(topic))
                topic.clear()
            topic.append('{0}。'.format(text))
        else:
            if len(topic) > 0:
                topic.append(text)
    if len(topic) > 0:
        topics.append(''.join(topic))
    return topics


def html2topic_all(r_dir):
    """
    指定されたパス以下にある yuhoディレクトリを取得する。
    yuhoディレクトリ下にある年度ディレクトリを取得する。
    各年度ディレクトリ下の全htmlファイルをテキストファイルに変換する。
    変換したテキストファイルを '指定されたパス/text'フォルダ下に保存する。

    Args:
        r_dir (str): 読み込みディレクトリのパス
    """
    # yuhoディレクトリ
    yuho_dir = os.path.join(r_dir, 'yuho')
    # テキスト保存ディレクトリが存在しない場合は作成する。
    text_dir = os.path.join(r_dir, 'topics') 
    if not os.path.isdir(text_dir): os.mkdir(text_dir)
    # yuhoディレクトリ下の各年度ディレクトリ名のリストを取得
    # dirpaths = os.listdir(yuho_dir)
    dirpaths = ['2020']
    # 全htmlファイルパスを取得
    filepaths = []
    for d in dirpaths:
        year_dir = os.path.join(yuho_dir, d)
        f = utils.get_filepaths(year_dir)
        for filepath in f:
            filepaths.append(filepath)
    # htmlファイルパスをテキストファイルに変換し、テキストファイルに保存
    for f in filepaths:
        # htmlファイルからテキストを取得
        topics = html2topic(f)
        if len(topics) > 0:
            # 企業ディレクトリを作成
            basename = os.path.basename(f)  # 読み込みファイル名を取得
            name = os.path.splitext(basename)[0]  # htm拡張子を削除
            code = re.split('_', name)  # 株式コードを取得
            y_dir = os.path.dirname(f).replace('yuho', 'topics')  # 年度ディレクトリの取得
            f_dir = os.path.join(y_dir, code[1])
            # 年度ディレクトリがなければ作成
            if not os.path.isdir(y_dir): os.mkdir(y_dir)
            # 企業ディレクトリがなければ作成
            if not os.path.isdir(f_dir): os.mkdir(f_dir)
            # ファイルに書き込み
            for i, text in enumerate(topics):
                w_file = os.path.join(f_dir, str(i) + '.txt')  # 書き込みファイル名
                with open(w_file, mode='w', encoding='utf-8') as f:
                    f.write(text)

if __name__ == '__main__':
    
    r_dir = './data'
    html2topic_all(r_dir)

    ################ test
    ###### html2topic
    # r_file = './data/sample/yuho/2020/10014_1436_yuho_101_20190430.htm'
    # texts = html2topic(r_file)
    # for text in texts:
    #     print('---------------------------------')
    #     print(text)
    ###### html2topic_all
    # r_dir = './data/sample'
    # html2topic_all(r_dir)