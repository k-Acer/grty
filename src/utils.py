import os

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