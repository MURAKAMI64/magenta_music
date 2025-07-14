### 感情ベクトル生成 環境構築・実行手順
---
```bash
conda create -n magenta python==3.7.9
```
```bash
conda activate magenta
```

```bash
pip install -r requirements.txt
```

```bash
python create_emotion_vectors.py
```

自分のPCで実行すると"The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine."という表示が出てしまいます。
