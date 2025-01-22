import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
import torch.nn.functional as F  
import torchvision.models as models  # Import this line to access pre-trained models
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# --------- 3. Dataset and Dataloader ---------
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


# --------- 2. Preprocess Data ---------

class DataPreprocessor:
    def __init__(self, prefix="train_HT_utterance_merged_part_"):
        self.prefix = prefix
        self.scaler = StandardScaler()


    # --------- 1. Merge CSV Files ---------
    def merge_csv_files(self,directory, output_file):
        csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(self.prefix) and f.endswith('.csv')]
        df_list = [pd.read_csv(file) for file in csv_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(os.path.join(directory, output_file), index=False)
        return merged_df
    
    def preprocess_data(self, df):
        file_names = df.iloc[:, 0]
        labels = df.iloc[:, -1].values
        print("Labels distribution: ", np.unique(labels, return_counts=True))
        features = df.iloc[:, 1:-1].values
        
        features_normalized = self.scaler.fit_transform(features)
        
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(features_normalized, labels, test_size=0.4, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return features_normalized, labels

    def create_dataloaders(self, batch_size=64):
        train_dataset = AudioDataset(self.X_train, self.y_train)
        val_dataset = AudioDataset(self.X_val, self.y_val)
        test_dataset = AudioDataset(self.X_test, self.y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

# --------- 4. Attention Model ---------
class AttentionModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes=2, num_heads=8):
        super(AttentionModel, self).__init__()

        # Embedding layer to transform input features into the desired embedding dimension
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Multi-Head Self Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # Feed Forward Network after Attention
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, num_classes)

        # Layer Normalization for stability
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Add sequence length dimension
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm(attn_output + x)
        x = F.relu(self.fc1(x))
        embeddings = x.squeeze(1)  # Squeeze sequence length dimension
        logits = self.fc2(self.dropout(embeddings))
        return logits, embeddings, attn_weights


# --------- 5. NT-Xent Loss ---------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature
        labels = torch.arange(z_i.size(0)).cuda()
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        return loss

class SpecnetTrainer:
    def __init__(self, model, criterion, optimizer, device, extract_utterance_level=True):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.extract_utterance_level = extract_utterance_level
        self.cross_entropy = nn.CrossEntropyLoss()

    def evaluate(self, val_loader):
        self.model.eval()  # Set model to evaluation mode
        all_labels = []
        all_preds = []
        
        with torch.no_grad():  # No gradients needed for evaluation
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # logits, _ = model(x)  # Forward pass
                logits, embeddings, _ = self.model(x)
                preds = torch.softmax(logits, dim=1)[:, 1]  # Assuming binary classification, take the positive class
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        all_labels = np.array(all_labels)
        print(f'Labels: {np.unique(all_labels)}')
        all_preds = np.array(all_preds)

        # Calculate AUC
        auc_score = roc_auc_score(all_labels, all_preds)

        # Calculate Confusion Matrix
        cm = confusion_matrix(all_labels, (all_preds > 0.5).astype(int))

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        eer = self.calculate_eer(fpr, tpr)

        return auc_score, cm, eer


    def train(self, train_loader, val_loader=None, epochs=50):

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                # logits, _ = model(x)
                logits, embeddings, _ = self.model(x)

                # Compute loss
                loss = self.cross_entropy(logits, y)  # Using cross-entropy for classification
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            if val_loader:
                # Evaluate model on validation set after each epoch
                auc_score, cm, eer = self.evaluate(val_loader)
                print(f"Validation AUC: {auc_score:.4f}, EER: {eer:.4f}")

    def calculate_eer(self, fpr, tpr):
        # Calculate EER (Equal Error Rate) from the FPR and TPR
        eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))  # Find where the FPR and TPR are closest
        eer = (fpr[eer_index] + (1 - tpr[eer_index])) / 2
        return eer

class SpecnetUtils:
    def __init__(self, model):
        self.model = model

    def save_model(self, save_dir, weights_name="specnet_weights.pth"):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, weights_name))

    def generate_embeddings(self, feature_Df, emb_output_path):

        # Load your trained model
        # model = torch.load('C:\Notebooks\rrl_source\Spectnet_model_Halftruth_embedding')
        self.model.eval()  # Set the model to evaluation mode for consistent embeddings

        # Read input CSV containing Fileid, raw features, and label
        # input_csv = 'specnet_embeddings.csv'  # Replace with the actual file path
        # df = pd.read_csv(input_csv)

        # Assuming the first column is 'Fileid', the last column is 'Label', and the intermediate columns are raw features
        # Get the first and last columns to drop
        fileids = feature_Df.iloc[:, 0].values  # First column
        labels = feature_Df.iloc[:, -1].values  # Last column

        # Drop the first and last columns
        raw_features = feature_Df.iloc[:, 1:-1].values  # All columns except the first and last

        # Convert raw features to a tensor
        raw_features_tensor = torch.tensor(raw_features, dtype=torch.float32).cuda()

        # Lists to store embeddings, file IDs, and labels
        all_embeddings = []
        all_fileids = []
        all_labels = []

        # Extract embeddings
        with torch.no_grad():  # No gradient tracking during extraction
            for i in range(len(raw_features)):
                feature = raw_features_tensor[i:i+1]  # Extract single feature (1, feature_size)
                label = labels[i]  # The corresponding label for the feature
                fileid = fileids[i]  # The corresponding file ID

                # Pass through the model to get the embeddings
                x = self.model.embedding(feature)  # Embedding layer
                x = x.unsqueeze(1)  # Add sequence length dimension
                attn_output, _ = self.model.attention(x, x, x)  # Attention layer
                embeddings = self.model.layer_norm(attn_output + x)  # Normalized attentive embeddings

                # Convert embeddings to CPU and numpy for storage
                embeddings_cpu = embeddings.cpu().numpy().flatten()  # Flatten to 1D if necessary

                # Append fileid, label, and embeddings to respective lists
                all_fileids.append(fileid)
                all_labels.append(label)
                all_embeddings.append(embeddings_cpu)

        # Convert lists to numpy arrays
        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)

        # Create a new DataFrame to save the extracted embeddings and labels
        columns = ['FileId'] + [f'Feature_{i+1}' for i in range(all_embeddings.shape[1])] + ['Label']
        output_df = pd.DataFrame(np.column_stack([all_fileids, all_embeddings, all_labels]), columns=columns)

        # Save the result to a new CSV file
        # output_csv = 'halftruth_attentive_embeddings.csv'  # Replace with desired output file path
        output_df.to_csv(emb_output_path, index=False)

        print(f"Extracted attentive embeddings saved to {emb_output_path}")


    def plot_tsne(self, features, labels, title, save_path, plot_type='features'):
            # Use torch.float32 dtype for input
        if plot_type == 'embeddings':
            with torch.no_grad():
                _,final_embeddings, _ = self.model(torch.tensor(features, dtype=torch.float32).cuda())  # Get only the embeddings (ignore attention_weights)
                features = final_embeddings.cpu().numpy()  # Convert embeddings to CPU for visualization
                title = "t-SNE After Training"

        tsne = TSNE(n_components=2, random_state=42)
        tsne_features = tsne.fit_transform(features)
        plt.figure(figsize=(8, 8))
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='viridis', s=10)
        plt.colorbar()
        plt.title(title)
        plt.savefig(save_path)
        plt.show()


# Merge the files
features_dir = "./../features"
emb_save_dir = "./../embeddings"
os.makedirs(emb_save_dir, exist_ok=True)  # Create directory if it doesn't exist
model_save_dir = "./../weights"
os.makedirs(model_save_dir, exist_ok=True)  # Create directory if it doesn't exist
output_file = "merged_features.csv"
data_preprocessor = DataPreprocessor()
# merged_DF = data_preprocessor.merge_csv_files(features_dir, output_file)
# print(f"Merged data shape: {merged_DF.shape}")
merged_DF = pd.read_csv(os.path.join(features_dir, output_file))   # temporary addition to read the merged file
features, labels = data_preprocessor.preprocess_data(merged_DF)
train_loader, val_loader, test_loader = data_preprocessor.create_dataloaders()

# --------- 6. Training and Evaluation ---------
input_dim = 1084
embedding_dim = 128
num_classes = 2
# Instantiate the model
specnet_model = AttentionModel(input_dim=input_dim, embedding_dim=embedding_dim, num_classes=num_classes).cuda()

criterion = NTXentLoss()
optimizer = optim.Adam(specnet_model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training the model with validation evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
specnet_trainer = SpecnetTrainer(specnet_model, criterion, optimizer, device)
specnet_trainer.train(train_loader, val_loader, epochs=10)  #temporary using train_loader for validation
specnet_utils = SpecnetUtils(specnet_model)
specnet_utils.save_model(model_save_dir, weights_name="specnet_weights.pth")
emb_save_path = os.path.join(emb_save_dir, 'specnet_emb.csv')
specnet_utils.generate_embeddings(merged_DF,emb_save_path)

# Load the model and generate embeddings
specnet_model = AttentionModel(input_dim=input_dim, embedding_dim=embedding_dim, num_classes=num_classes).cuda()
specnet_model.load_state_dict(torch.load(os.path.join(model_save_dir, "specnet_weights.pth")))
specnet_utils = SpecnetUtils(specnet_model)
specnet_utils.generate_embeddings(merged_DF,emb_save_path)
specnet_utils.plot_tsne(features[:80000], labels[:80000], "t-SNE Before Training", "tsne_before.png", plot_type='features')
specnet_utils.plot_tsne(features[:80000], labels[:80000], "t-SNE After Training", "tsne_after.png", plot_type='embeddings')