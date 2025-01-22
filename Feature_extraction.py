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
import argparse


class FeatureExtractor:
    def __init__(self, wav2vec_processor, wave2vec_model, device, extract_utterance_level=True):
        self.wav2vec_processor = wav2vec_processor
        self.wave2vec_model = wave2vec_model.to(device)
        self.device = device
        self.window_size = 0.16  # Segment length in seconds
        self.hop_size = 0.15     # Frame shift in seconds
        self.extract_utterance_level = extract_utterance_level  # Set to True for utterance-level, False for segment-level

        # Main processing function
    def extract_features(self, audio_files, labels):
        all_features = []
        for audio_file in tqdm(audio_files, desc="Processing files"):
            file_id = os.path.basename(audio_file).replace('.wav', '')
            if file_id not in labels:
                continue

            y, sr = librosa.load(audio_file, sr=16000)
            if self.extract_utterance_level:
                label = labels[file_id]
                handcrafted_features = self.extract_handcrafted_features(y, sr)
                wav2vec_features = self.extract_wav2vec_features(y, sr)
                combined_features = np.concatenate(([file_id], handcrafted_features, wav2vec_features, [label]))
                all_features.append(combined_features)
            else:
                segment_labels = labels[file_id]
                segment_length = int(self.window_size * sr)
                hop_length = int(self.hop_size * sr)

                for i, seg_label in enumerate(segment_labels):
                    start = i * hop_length
                    end = start + segment_length
                    segment = (
                        np.pad(y[start:], (0, segment_length - len(y[start:])), mode='edge')
                    if end > len(y)
                    else y[start:end]
                    )

                    # Extract features
                    handcrafted_features = self.extract_handcrafted_features(segment, sr)
                    wav2vec_features = self.extract_wav2vec_features(segment, sr)
                    combined_features = np.concatenate(([file_id], handcrafted_features, wav2vec_features, [seg_label]))
                    all_features.append(combined_features)

        return np.array(all_features)


    def extract_wav2vec_features(self, segment, sr): 
        if len(segment.shape) > 1:   # Ensure the segment is a 1D array and not a 2D array
            segment = segment.flatten()  # Flatten it to 1D if needed
        
        inputs = self.wav2vec_processor(segment, sampling_rate=sr, return_tensors="pt", padding=False)
        inputs = {key: value.to(device) for key, value in inputs.items()} 
        
        with torch.no_grad():
            outputs = self.wave2vec_model(**inputs)  # Extract the last hidden state and return it
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Hann window function
    def hann_window(self, length):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(length) / (length - 1)))

    # Handcrafted feature extraction
    def extract_handcrafted_features(self, segment, sr):
        windowed_segment = segment * self.hann_window(len(segment))
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


class ProcessAudios:
    def __init__(self, data_root, train_audio_path, segment_label_path, utterance_label_path, extract_utterance_level, save_path, device):
        self.data_root = data_root
        self.train_audio_path = train_audio_path
        self.segment_label_path = segment_label_path
        self.utterance_label_path = utterance_label_path
        self.extract_utterance_level = extract_utterance_level
        self.save_path = save_path
        self.device = device

    def process(self):
        print("Arguments received:")
            # Load labels and process dataset
        labels = self.load_labels(self.utterance_label_path if self.extract_utterance_level else self.segment_label_path, self.extract_utterance_level)
        audio_files = glob(self.train_audio_path)
        # audio_files = audio_files[:500]
        wav2vec_processor, wave2vec_model = self.load_models()
        feature_extractor = FeatureExtractor( wav2vec_processor, wave2vec_model, self.device, self.extract_utterance_level)
        features = feature_extractor.extract_features(audio_files, labels)

        # Save features in parts
        part_size = 10000
        for i in range(0, len(features), part_size):
            part_features = features[i:i + part_size]
            part_name = f"{self.save_path}train_{'HT_utterance' if extract_utterance_level else 'segment'}_merged_part_{i // part_size + 1}.csv"
            pd.DataFrame(part_features).to_csv(part_name, index=False)
        
        return features
        

    def load_models(self):   
        # Initialize paths and parameters
        model_name = "facebook/wav2vec2-large-xlsr-53"
        wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wave2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        # model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        return wav2vec_processor, wave2vec_model



    # Load segment-level or utterance-level labels
    def load_labels(self, label_file, utterance_level):
        if utterance_level:
            labels = {}
            with open(label_file, 'r') as file:
                for line in file:
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        file_id = parts[0].strip()
                        label = parts[-1].strip()
                        labels[file_id] = 0 if label == '0' else 1
        else:
            labels = np.load(label_file, allow_pickle=True).item()
        return labels



if __name__ == "__main__":
    data_root = "/mnt/f/Awais_data/Datasets/Halftruth/HAD_train"
    train_audio_path = os.path.join(data_root, "train/conbine/*.wav")
    segment_label_path = os.path.join(data_root, "HAD_train_label.txt")
    utterance_label_path = os.path.join(data_root, "HAD_train_label.txt")
    features_save_path = "./../features/"
    os.makedirs(features_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_utterance_level=True

    proecss_audios = ProcessAudios(data_root, train_audio_path, segment_label_path, utterance_label_path, extract_utterance_level, features_save_path, device)
    features = proecss_audios.process()
    print(f"Features extracted and saved in {features_save_path}; Features shape: {len(features)}")



