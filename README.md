# grty (Get Risk Topiks from Yuho)

有価証券報告書からLDAを用いてリスクトピックを取得するプログラム。

## 分析の流れ
htm 形式の文書から html タグを消去し、txt 形式に変換する。
Juman++ を用いて形態素解析
LDA
結果をcsv形式で出力
