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
    filelist = []
    for f in os.listdir(r_dir):
        if os.path.isfile(os.path.join(r_dir, f)):
            filelist.append(r_dir + f)

    return filelist