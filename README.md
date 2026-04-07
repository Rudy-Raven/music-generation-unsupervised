# Music Generation using Unsupervised Machine Learning

CSE425 Project - Task 1

## Overview
This project implements an LSTM Autoencoder for unsupervised music generation using MIDI data.

## Pipeline
- MIDI preprocessing (PrettyMIDI)
- Piano-roll conversion
- Sequence generation (64x128)
- LSTM Autoencoder training
- Music generation

## Project Structure
- src/preprocessing → data processing
- src/models → LSTM autoencoder
- src/training → training script
- src/generation → music generation

## Notes
- Large datasets and outputs are excluded via .gitignore
- Model trained on subset of MAESTRO dataset
