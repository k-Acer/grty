import os

from bs4 import BeautifulSoup


def html2text(r_file, w_dir):
    """
    htm(html)ファイルからテキストデータを取得し、
    テキストデータをテキストファイルに保存する。

    Args:
        r_file (str): HTMLファイルが格納されたファイルのパス
        w_dir (str): テキストファイルを保存するディレクトリのパス
    """
    # テキストの読み込み
    with open(r_file) as f:
        html = f.read()

    # テキストの取得
    soup = BeautifulSoup(html,"html.parser")
    text = soup.get_text()

    # テキストの書き込み
    basename = os.path.basename(r_file)  # 読み込みファイル名を取得
    name = os.path.splitext(basename)[0]  # 拡張子を削除
    w_file = w_dir + name + '.txt'
    with open(w_file, mode='w') as f:
        f.write(text)

def roop_html2text(r_dir, w_dir):
    """
    html2text を繰り返し実行する。

    Args:
        r_dir (str): HTMLファイルが格納されたディレクトリのパス
        w_dir (str): テキストファイルを保存するディレクトリのパス
    """
    # 各HTMLファイルからテキストを取得し、保存する。
    filepaths = get_filepaths(r_dir)
    for r_file in filepaths:
        html2text(r_file, w_dir)

def get_filepaths(r_dir):
    """
    r_dir 以下のファイル名を全て取得する。

    Args:
        r_dir (str): 読み込みディレクトリのパス

    Returns:
        list: 各ファイルのパスをリストに格納して返す
    """
    # テキストのファイル名を取得
    filenames = os.listdir(r_dir)
    filepaths = [r_dir + f for f in filenames]
    return filepaths

if __name__ == '__main__':

    r_dir = './data/yuho/2019/'
    w_dir = './data/text/2019/'
    roop_html2text(r_dir, w_dir)
