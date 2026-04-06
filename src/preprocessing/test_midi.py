import os
import pretty_midi

midi_dir = "data/processed/sample_midi"

files = [f for f in os.listdir(midi_dir) if f.endswith(".midi") or f.endswith(".mid")]

print("Total MIDI files found:", len(files))

if len(files) == 0:
    print("No MIDI files found!")
    exit()

sample_path = os.path.join(midi_dir, files[0])
print("Testing file:", sample_path)

midi = pretty_midi.PrettyMIDI(sample_path)

print("Number of instruments:", len(midi.instruments))

total_notes = sum(len(inst.notes) for inst in midi.instruments)
print("Total notes:", total_notes)
