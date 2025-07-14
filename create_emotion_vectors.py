print("os, numpyをインポートします")
import os
import numpy as np
print("magentaをインポートします")
import magenta
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

# --- 1. モデルと定数の設定 ---

print("モデルと定数の設定をします")
# 使用するMusicVAEの学習済みモデルを指定
# ここではメロディ用の一般的なモデル 'cat-mel_2bar_big' を使用します
MODEL_NAME = 'cat-mel_2bar_big'
config = configs.CONFIG_MAP[MODEL_NAME]
model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=f'./{MODEL_NAME}.ckpt')

# MIDIデータセットの親フォルダへのパス
MIDI_BASE_PATH = 'emotion_midi/'
# 処理する感情のリスト
EMOTIONS = ['happy', 'sad', 'angry', 'surprised']


# --- 2. 各感情の平均ベクトルを計算 ---

print("感情ベクトルの計算を開始します...")
emotion_vectors = {}

for emotion in EMOTIONS:
    print(f"'{emotion}'のベクトルを計算中...")

    emotion_midi_dir = os.path.join(MIDI_BASE_PATH, emotion)
    midi_files = [os.path.join(emotion_midi_dir, f) for f in os.listdir(emotion_midi_dir) if f.endswith(('.mid', '.midi'))]

    # この感情に属するすべての曲のベクトルを保存するリスト
    vectors_for_this_emotion = []

    for midi_file in midi_files:
        try:
            # MIDIファイルをNoteSequenceオブジェクトに変換
            note_sequence = mm.midi_file_to_note_sequence(midi_file)
            
            # NoteSequenceをモデルが扱えるTensorに変換し、ベクトル化（エンコード）
            # encodeはリストで返すが、短い曲なので最初の要素を取得
            z, _, _ = model.encode([note_sequence])
            vectors_for_this_emotion.append(z[0])
            
        except Exception as e:
            print(f"ファイル {midi_file} の処理中にエラー: {e}")

    if vectors_for_this_emotion:
        # NumPyを使って全ベクトルの平均を計算
        average_vector = np.mean(vectors_for_this_emotion, axis=0)
        emotion_vectors[emotion] = average_vector
        print(f"'{emotion}'の平均ベクトルが計算されました。")

print("\n--- 全ての平均感情ベクトルが準備完了 ---")
# 計算されたベクトルの情報を表示（オプション）
for emotion, vector in emotion_vectors.items():
    print(f"Emotion: {emotion}, Vector Shape: {vector.shape}")


# --- 3. 表情パラメータに基づいて潜在空間ベクトルを合成 ---

# 例：顔認識システムから得られた表情の確率
# この値はリアルタイムで変動することを想定
emotion_probabilities = {
    "happy": 0.80,
    "sad": 0.02,
    "angry": 0.03,
    "surprised": 0.15
}

# 合成後のベクトルを初期化（ベクトルの次元数は同じなので、どれか一つを代表として使う）
# np.zerosで初期化してもOK
combined_vector = np.zeros(model.latent_dim)

print("\n--- 感情の確率に基づいてベクトルを合成します ---")
for emotion, probability in emotion_probabilities.items():
    if emotion in emotion_vectors:
        # 各感情ベクトルにその確率（重み）を掛けて足し合わせる
        weighted_vector = emotion_vectors[emotion] * probability
        combined_vector += weighted_vector
        print(f"'{emotion}'のベクトルを {probability*100}% の重みで加算しました。")

print("\n--- 合成ベクトルが完成しました ---")
print("Final Combined Vector (最初の10次元のみ表示):", combined_vector[:10])
print("Vector Shape:", combined_vector.shape)


# --- 4. 次のステップ：音楽の生成 ---
# この 'combined_vector' をMusicVAEのデコーダに入力することで、
# 感情がミックスされた新しい音楽を生成できます。
#
# 例：
# generated_sequences = model.decode(z=np.array([combined_vector]), length=256) # lengthは音符の数
# mm.note_sequence_to_midi_file(generated_sequences[0], 'output_music.mid')
# print("\n音楽ファイル 'output_music.mid' が生成されました。")