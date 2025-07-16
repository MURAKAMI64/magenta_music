### 感情ベクトル生成 環境構築・実行手順
---
```bash
conda create -n magenta python==3.7.9
```
```bash
conda activate magenta
```

```bash
pip install magenta pyfluidsynth pretty_midi
```

```bash
python create_emotion_vectors.py
```

自分のPCで実行するとmagentaライブラリをインポートする時に"The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine."という表示が出てしまい、それ以降のコードが実行されません。
