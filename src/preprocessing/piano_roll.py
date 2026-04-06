import os
import numpy as np
import pretty_midi

MIDI_DIR = "data/processed/sample_midi"
OUTPUT_DIR = "data/processed/piano_rolls"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def midi_to_piano_roll(midi_path, fs=8):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)

    piano_roll = (piano_roll > 0).astype(np.float32)
    piano_roll = piano_roll.T

    return piano_roll

def main():
    files = [f for f in os.listdir(MIDI_DIR) if f.endswith(".mid") or f.endswith(".midi")]

    print(f"Found {len(files)} MIDI files")

    for i, filename in enumerate(files):
        path = os.path.join(MIDI_DIR, filename)

        try:
            piano_roll = midi_to_piano_roll(path)

            save_name = os.path.splitext(filename)[0] + ".npy"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            np.save(save_path, piano_roll)

            print(f"[{i+1}/{len(files)}] Saved: {save_name}, shape={piano_roll.shape}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()