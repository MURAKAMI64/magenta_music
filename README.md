# 感情ベクトル生成 環境構築・実行手順

このドキュメントは、感情ベクトルを生成するためのPython環境を構築し、スクリプトを実行する手順をまとめたものです。
`conda`がインストールされていることを前提としています。

---

### ステップ1: Conda仮想環境の作成

まず、プロジェクト用に独立した仮想環境を作成します。これにより、他のプロジェクトとの依存関係の衝突を防ぎます。

`magenta`という名前で、Pythonバージョン3.7.9の仮想環境を作成します。

```bash
conda create -n magenta python==3.7.9

conda activate magenta

pip install -r requirements.txt

python create_emotion_vectors.py
