import os

from bs4 import BeautifulSoup
import chardet

import utils


def html2text(r_file, w_dir):
    """
    htm(html)ファイルからテキストデータを取得し、
    テキストデータをテキストファイルに保存する。

    Args:
        r_file (str): HTMLファイルが格納されたファイルのパス
        w_dir (str): テキストファイルを保存するディレクトリのパス
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
    soup = BeautifulSoup(html,"html.parser")
    text = soup.get_text()
    # 書き込みディレクトリが存在しない場合は作成
    if not os.path.isdir(w_dir): os.mkdir(w_dir)
    # テキストの書き込み
    basename = os.path.basename(r_file)  # 読み込みファイル名を取得
    name = os.path.splitext(basename)[0]  # 拡張子を削除
    w_path = w_dir + name + '.txt'
    with open(w_path, mode='w', encoding='utf-8') as f:
        f.write(text)

def roop_html2text(r_dir, w_dir):
    """
    html2text を繰り返し実行する。

    Args:
        r_dir (str): HTMLファイルが格納されたディレクトリのパス
        w_dir (str): テキストファイルを保存するディレクトリのパス
    """
    # フォルダの存在を確認
    if os.path.isdir(r_dir):
        # ファイルパスの取得
        filepaths = utils.get_filepaths(r_dir)
        # 各HTMLファイルからテキストを取得し、保存する。
        if len(filepaths) == 0:
            print('ファイルが存在しません')
        else:
            for r_file in filepaths:
                html2text(r_file, w_dir)
    else:
        print('{0} というフォルダは存在しません'.format(r_dir))

if __name__ == '__main__':
    
    r_dir = './data/yuho/2016/'
    w_dir = './data/text/2016/'
    roop_html2text(r_dir, w_dir)

    # sample
    # r_dir = './data/yuho_sample/2019/'
    # w_dir = './data/text_sample/2019/'
    # roop_html2text(r_dir, w_dir)
