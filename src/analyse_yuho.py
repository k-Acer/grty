import MeCab


def analyse_by_mecab(r_path):
    """
    テキストファイルに対して形態素解析を行う。
    名詞のみを抽出し、リストに格納する。

    Args:
        r_path (str): 読み込みファイル名

    Returns:
        list: 形態素解析を行った単語をリスト型で返す
    """
    # テキストの読み込み
    with open(r_path) as f:
        text = f.read()
    # 形態素解析
    m = MeCab.Tagger ("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    node = m.parseToNode(text)
    word_list = []
    # 名詞の格納
    while node:
        word = node.surface
        hinshi = node.feature.split(",")[0]
        if hinshi == '名詞':
            word_list.append(word)
        node = node.next
    
    return word_list

def roop_analyse():
    """
    analyse_by_mecab を繰り返し実行する。

    Args:

    Returns:
    """
    print('roop')


if __name__ == '__main__':

    idemitu = analyse_by_mecab('./data/text/idemitu.txt')
    takeda = analyse_by_mecab('./data/text/takeda.txt')
    toyota = analyse_by_mecab('./data/text/toyota.txt')

    print(takeda)
    