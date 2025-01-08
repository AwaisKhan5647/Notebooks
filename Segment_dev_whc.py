#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Initialize paths and parameters
model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Vec2Model.from_pretrained(model_name).to(device)

dev_audio_path = "/gpfs/accounts/drmalik_root/drmalik0/mawais/PartialSpoof/dev/con_wav/*.wav"
segment_label_path = "/gpfs/accounts/drmalik_root/drmalik0/mawais/PartialSpoof/database_segment_labels/database/segment_labels/dev_seglab_0.16.npy"
utterance_label_path = "/gpfs/accounts/drmalik_root/drmalik0/mawais/PartialSpoof/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt"
save_path = "/gpfs/accounts/drmalik_root/drmalik0/mawais/PartialSpoof/Specnet/dev/"

window_size = 0.16  # Segment length in seconds
hop_size = 0.16     # Frame shift in seconds
extract_utterance_level = False  # Set to True for utterance-level, False for segment-level

# Hann window function
def hann_window(length):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))

# Load segment-level or utterance-level labels
def load_labels(label_file, utterance_level):
    if utterance_level:
        labels = {}
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) >= 5:
                    file_id = parts[1].strip()
                    label = parts[-1].strip()
                    labels[file_id] = 0 if label == 'spoof' else 1
    else:
        labels = np.load(label_file, allow_pickle=True).item()
    return labels

# Handcrafted feature extraction
def extract_handcrafted_features(segment, sr):
    windowed_segment = segment * hann_window(len(segment))
    mfcc = librosa.feature.mfcc(y=windowed_segment, sr=sr, n_mfcc=13).mean(axis=1)
    delta_mfcc = librosa.feature.delta(mfcc)
    tempo = librosa.beat.tempo(y=windowed_segment, sr=sr)[0]
    chroma = librosa.feature.chroma_stft(y=windowed_segment, sr=sr).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(windowed_segment).mean()
    energy = librosa.feature.rms(y=windowed_segment).mean()
    pitches, _ = librosa.core.piptrack(y=windowed_segment, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
    tempogram = librosa.feature.tempogram(y=windowed_segment, sr=sr).mean(axis=1)[1:]
    downsampled_tempogram = tempogram[::int(np.ceil(len(tempogram) / 18))]
    features = np.concatenate((mfcc, delta_mfcc, [tempo], chroma, [zcr], [energy], [pitch], downsampled_tempogram))
    return features

def extract_wav2vec_features(segment, sr): 
    if len(segment.shape) > 1:   # Ensure the segment is a 1D array and not a 2D array
        segment = segment.flatten()  # Flatten it to 1D if needed
    
    inputs = feature_extractor(segment, sampling_rate=sr, return_tensors="pt", padding=False)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    
    with torch.no_grad():
        outputs = model(**inputs)  # Extract the last hidden state and return it
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()



# Main processing function
def process_dataset(audio_files, labels, utterance_level):
    all_features = []
    for audio_file in tqdm(audio_files, desc="Processing files"):
        file_id = os.path.basename(audio_file).replace('.wav', '')
        if file_id not in labels:
            continue

        y, sr = librosa.load(audio_file, sr=16000)
        if utterance_level:
            label = labels[file_id]
            handcrafted_features = extract_handcrafted_features(y, sr)
            wav2vec_features = extract_wav2vec_features(y, sr)
            combined_features = np.concatenate(([file_id], handcrafted_features, wav2vec_features, [label]))
            all_features.append(combined_features)
        else:
            segment_labels = labels[file_id]
            segment_length = int(window_size * sr)
            hop_length = int(hop_size * sr)

            for i, seg_label in enumerate(segment_labels):
                start = i * hop_length
                end = start + segment_length
                segment = (
                    np.pad(y[start:], (0, segment_length - len(y[start:])), mode='edge')
                if end > len(y)
                else y[start:end]
                )

                # Extract features
                handcrafted_features = extract_handcrafted_features(segment, sr)
                wav2vec_features = extract_wav2vec_features(segment, sr)
                combined_features = np.concatenate(([file_id], handcrafted_features, wav2vec_features, [seg_label]))
                all_features.append(combined_features)

    return np.array(all_features)



# Load labels and process dataset
labels = load_labels(utterance_label_path if extract_utterance_level else segment_label_path, extract_utterance_level)
audio_files = glob(dev_audio_path)
#audio_files = audio_files[:1]
features = process_dataset(audio_files, labels, extract_utterance_level)

# Save features in parts
part_size = 90000
for i in range(0, len(features), part_size):
    part_features = features[i:i + part_size]
    part_name = f"{save_path}dev_{'utterance' if extract_utterance_level else 'segment'}_merged_part_{i // part_size + 1}.csv"
    pd.DataFrame(part_features).to_csv(part_name, index=False)






