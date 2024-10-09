import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, k):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.k = k

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        kmers = [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
        kmers_str = " ".join(kmers)
        encoded = self.tokenizer(kmers_str, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze(), label

def prepare_data(file_path, k):
    df = pd.read_csv(file_path)
    df = df.sample(frac=0.05, random_state=42)  # Randomly select 5% of the data

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['organism_part'])

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("zhihan1996/DNA_bert_6", do_lower_case=False)

    train_dataset = DNASequenceDataset(train_df['seq'].values, train_df['label'].values, tokenizer, k)
    val_dataset = DNASequenceDataset(val_df['seq'].values, val_df['label'].values, tokenizer, k)

    return train_dataset, val_dataset, len(le.classes_)

def train_model(train_loader, val_loader, num_labels, num_epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained("zhihan1996/DNA_bert_6", num_labels=num_labels)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    file_path = 'data/dna_sequences.csv'
    k = 6  # k-mer size

    train_dataset, val_dataset, num_labels = prepare_data(file_path, k)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = train_model(train_loader, val_loader, num_labels)
    torch.save(model.state_dict(), 'models/dna_classification_model.pth')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
