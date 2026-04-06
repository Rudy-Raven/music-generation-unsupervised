import os
import numpy as np

PIANO_ROLL_DIR = "data/processed/piano_rolls"
OUTPUT_PATH = "data/processed/train_sequences.npy"

SEQ_LEN = 64
STRIDE = 16

def create_sequences(arr, seq_len=64, stride=16):
    sequences = []

    if len(arr) < seq_len:
        return sequences

    for i in range(0, len(arr) - seq_len + 1, stride):
        seq = arr[i:i + seq_len]
        if seq.shape == (seq_len, 128):
            sequences.append(seq)

    return sequences

def main():
    files = [f for f in os.listdir(PIANO_ROLL_DIR) if f.endswith(".npy")]
    all_sequences = []

    print(f"Found {len(files)} piano roll files")

    for filename in files:
        path = os.path.join(PIANO_ROLL_DIR, filename)
        arr = np.load(path)
        seqs = create_sequences(arr, SEQ_LEN, STRIDE)
        all_sequences.extend(seqs)
        print(f"{filename}: {len(seqs)} sequences")

    all_sequences = np.array(all_sequences, dtype=np.float32)
    print("Final dataset shape:", all_sequences.shape)

    np.save(OUTPUT_PATH, all_sequences)
    print(f"Saved dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()