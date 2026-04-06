import os
import numpy as np
import torch
import pretty_midi

from src.models.lstm_autoencoder import LSTMAutoencoder

MODEL_PATH = "outputs/lstm_autoencoder.pth"
DATA_PATH = "data/processed/train_sequences.npy"
OUTPUT_DIR = "outputs/samples"   # ✅ changed here

SEQ_LEN = 64
LATENT_NOISE_STD = 0.05
THRESHOLD = 0.15
TIME_STEP = 0.03

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def load_data():
    return np.load(DATA_PATH).astype(np.float32)

def generate_from_real_latent(model, real_sequence):
    x = torch.tensor(real_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        z = model.encode(x)
        noise = torch.randn_like(z) * LATENT_NOISE_STD
        z_new = z + noise
        output = model.decode(z_new, SEQ_LEN)
        output = torch.sigmoid(output)

    return output.squeeze(0).numpy()

def piano_roll_to_midi(piano_roll, filename):
    piano_roll = (piano_roll > THRESHOLD).astype(int)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time_steps, pitches = piano_roll.shape

    for pitch in range(pitches):
        active = False
        start_time = 0.0

        for t in range(time_steps):
            note_on = piano_roll[t, pitch] > 0

            if note_on and not active:
                active = True
                start_time = t * TIME_STEP

            elif not note_on and active:
                end_time = t * TIME_STEP
                note = pretty_midi.Note(
                    velocity=90,
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
                active = False

        if active:
            end_time = time_steps * TIME_STEP
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(filename)

def main():
    model = load_model()
    data = load_data()

    rng = np.random.default_rng(42)

    indices = rng.choice(len(data), size=15, replace=False)

    for i, idx in enumerate(indices, start=1):
        generated = generate_from_real_latent(model, data[idx])
        filename = os.path.join(OUTPUT_DIR, f"sample_{i}.mid")  # ✅ clean naming
        piano_roll_to_midi(generated, filename)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()