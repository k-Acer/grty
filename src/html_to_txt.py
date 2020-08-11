import os

from bs4 import BeautifulSoup

def html2text(r_path):
    """
    htm(html)ファイルからテキストデータを取得し、
    テキストデータをテキストファイルに保存する。
    [この関数内でファイルの読み書きを行うのはあまり良くない気がする。]
    """
    # テキストの読み込み
    # windows環境では cp932によるエンコードがデフォルトなので、utf-8によるエンコードを行う。
    with open(r_path, encoding="utf-8") as f:
        html = f.read()

    # テキストの取得
    soup = BeautifulSoup(html,"html.parser")
    text = soup.get_text()

    # テキストの書き込み
    basename = os.path.basename(r_path)  # 読み込みファイル名を取得
    name = os.path.splitext(basename)[0]  # 拡張子を削除
    w_path = 'data\\text\\' + name + '.txt'
    with open(w_path, mode='w', encoding="utf-8") as f:
        f.write(text)


if __name__ == '__main__':
    r_path = 'data\\yuho\\takeda.htm'  # 読み込みファイル名
    html2text(r_path)
