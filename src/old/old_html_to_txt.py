import os
import re

from bs4 import BeautifulSoup
import chardet

import utils


def html2text(r_file):
    """
    htm(html)ファイルからテキストデータを取得する。

    Args:
        r_file (str): HTMLファイルが格納されたファイルのパス

    Returns:
        text (str): テキストデータ
    """
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
    text = soup.get_text()
    return text

def html2text_year(base_dir, year):
    """
    １年分

    Args:
        base_dir (str): ベースとなるパス
        year (str): 年度
    """
    yuho_dir = os.path.join(base_dir, 'yuho')
    # テキスト保存ディレクトリが存在しない場合は作成する。
    text_dir = os.path.join(base_dir, 'text') 
    if not os.path.isdir(text_dir): os.mkdir(text_dir)
    # yuhoディレクトリ下の各年度ディレクトリ名のリストを取得
    year_dir = os.path.join(yuho_dir, year)
    # 全htmlファイルパスを取得
    filepaths = []
    f = utils.get_filepaths(year_dir)
    for filepath in f:
        filepaths.append(filepath)
    # htmlファイルパスをテキストファイルに変換し、テキストファイルに保存
    for filepath in filepaths:
        # htmlファイルからテキストを取得
        text = html2text(filepath)
        # 書き込みファイル名の作成
        basename = os.path.basename(filepath)  # 読み込みファイル名を取得
        name = os.path.splitext(basename)[0]  # 拡張子を削除
        name_list = re.split('_', name)  # 株式コード
        w_dir = os.path.dirname(filepath).replace('yuho', 'text')  # 書き込みディレクトリ名を作成
        # 書き込みディレクトリが存在しない場合は作成する。
        if not os.path.isdir(w_dir): os.mkdir(w_dir)
        w_file = os.path.join(w_dir, name_list[1] + '.txt')  # 書き込みファイル名
        # テキストファイルに保存
        with open(w_file, mode='w', encoding='utf-8') as f:
            f.write(text)

def html2text_all(r_dir):
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
    text_dir = os.path.join(r_dir, 'text') 
    if not os.path.isdir(text_dir): os.mkdir(text_dir)
    # yuhoディレクトリ下の各年度ディレクトリ名のリストを取得
    dirpaths = os.listdir(yuho_dir)
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
        text = html2text(f)
        # 書き込みファイル名の作成
        basename = os.path.basename(f)  # 読み込みファイル名を取得
        name = os.path.splitext(basename)[0]  # 拡張子を削除
        name_list = re.split('_', name)  # 株式コード
        w_dir = os.path.dirname(f).replace('yuho', 'text')  # 書き込みディレクトリ名を作成
        # 書き込みディレクトリが存在しない場合は作成する。
        if not os.path.isdir(w_dir): os.mkdir(w_dir)
        w_file = os.path.join(w_dir, name_list[1] + '.txt')  # 書き込みファイル名
        # テキストファイルに保存
        with open(w_file, mode='w', encoding='utf-8') as f:
            f.write(text)
 


if __name__ == '__main__':
    # 全期間
    r_dir = './data'
    # html2text_all(r_dir)

    # １年間
    # base_dir = './data'
    # year = '2020'
    # html2text_year(base_dir, year)

    ################ sample
    # r_dir = './data/sample'
    # html2text_all(r_dir)
