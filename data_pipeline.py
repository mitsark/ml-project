"""
Mental Health Sentiment Analysis - Data Pipeline
Handles: Loading, Cleaning, Tokenization, DataLoader creation
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Current dataset class set (7 classes).
# Note: user message said "8 states" but listed 7 labels; this mapping follows
# the provided label list and the downloaded dataset values.
CURRENT_STATUS_TO_LABEL = {
    'Anxiety': 0,
    'Depression': 1,
    'Bipolar': 2,
    'Normal': 3,
    'Personality Disorder': 4,
    'Stress': 5,
    'Suicidal': 6,
}
CURRENT_CLASS_NAMES = [
    'Anxiety',
    'Depression',
    'Bipolar',
    'Normal',
    'Personality Disorder',
    'Stress',
    'Suicidal',
]

STATUS_NORMALIZATION = {
    'anxiety': 'Anxiety',
    'depression': 'Depression',
    'bipolar': 'Bipolar',
    'normal': 'Normal',
    'personality disorder': 'Personality Disorder',
    'personality disorder ': 'Personality Disorder',
    'personality disorder.': 'Personality Disorder',
    'personality disorder,': 'Personality Disorder',
    'stress': 'Stress',
    'suicidal': 'Suicidal',
}

# ============================================================================
# STAGE 1 & 2: LOAD AND EXPLORE DATA
# ============================================================================

def load_and_explore_data(csv_path):
    """Load dataset and generate exploration plots"""
    print("=" * 70)
    print("STAGE 1-2: DATA LOADING AND EXPLORATION")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_path)

    # Normalize supported schemas to a common internal format:
    # text (str) + label (int)
    if {'statement', 'status'}.issubset(df.columns):
        df['text'] = df['statement'].fillna('').astype(str)
        raw_status = df['status'].astype(str).str.strip()
        df['status'] = (
            raw_status
            .str.lower()
            .map(STATUS_NORMALIZATION)
        )

        unknown_statuses = sorted(raw_status[df['status'].isna()].unique().tolist())
        if unknown_statuses:
            raise ValueError(
                f"Unknown status values found: {unknown_statuses}. "
                f"Expected one of: {list(CURRENT_STATUS_TO_LABEL.keys())}"
            )

        df['label'] = df['status'].map(CURRENT_STATUS_TO_LABEL).astype(int)
        print("\nDetected schema: [statement, status] (optional id column supported)")
    elif {'text', 'label'}.issubset(df.columns):
        # Backward compatibility with older numeric-label datasets.
        print("\nDetected schema: [text, label]")
    else:
        raise ValueError(
            "Unsupported dataset schema. Expected columns [statement, status] "
            "or [text, label]."
        )

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic info
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Class distribution
    print(f"\nClass distribution (counts):")
    print(df['label'].value_counts().sort_index())
    
    print(f"\nClass distribution (percentages):")
    class_percentages = (df['label'].value_counts(normalize=True).sort_index() * 100).round(2)
    for label, pct in class_percentages.items():
        print(f"  Class {label}: {pct:5.2f}%")
    
    # Text length analysis
    df['text_length'] = df['text'].str.split().apply(len)
    print(f"\nText length statistics:")
    print(df['text_length'].describe())
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Class distribution
    df['label'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='steelblue', edgecolor='black'
    )
    axes[0].set_title('Class Distribution (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Class distribution percentage
    (df['label'].value_counts(normalize=True).sort_index() * 100).plot(
        kind='bar', ax=axes[1], color='coral', edgecolor='black'
    )
    axes[1].set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class Label')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Text length distribution
    axes[2].hist(df['text_length'], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[2].axvline(df['text_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["text_length"].mean():.0f}')
    axes[2].axvline(512, color='orange', linestyle='--', linewidth=2, label='RoBERTa limit: 512')
    axes[2].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Number of Tokens')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved exploration plots to 'data_exploration.png'")
    plt.close()
    
    # Sample examples from each class
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES FROM EACH CLASS")
    print("=" * 70)
    class_names = CURRENT_CLASS_NAMES
    
    for label in sorted(df['label'].unique()):
        sample_text = str(df[df['label'] == label]['text'].iloc[0])
        print(f"\nClass {label} ({class_names[label]}): Length={len(sample_text.split())} tokens")
        print(f"  Text: {sample_text[:150]}...")
    
    return df

# ============================================================================
# STAGE 3: TEXT CLEANING
# ============================================================================

def clean_text(text):
    """
    Clean text by:
    - Removing URLs
    - Removing @mentions
    - Removing special characters (keep only letters and spaces)
    - Normalizing whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    #text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove @mentions
    #text = re.sub(r'@\w+', '', text)
    
    # Remove special characters (keep letters, numbers, spaces, and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\-]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(df):
    """Apply cleaning to all texts"""
    print("\n" + "=" * 70)
    print("STAGE 3: TEXT PREPROCESSING")
    print("=" * 70)
    
    print("\nApplying text cleaning...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Show before/after
    print("\nBefore cleaning:")
    print(f"  Example: {str(df['text'].iloc[0])[:100]}...")
    print(f"\nAfter cleaning:")
    print(f"  Example: {df['text_clean'].iloc[0][:100]}...")
    
    # Check text lengths after cleaning
    df['text_clean_length'] = df['text_clean'].str.split().apply(len)
    print(f"\nText length after cleaning (word count):")
    print(df['text_clean_length'].describe())
    
    # Check for empty texts after cleaning
    empty_count = (df['text_clean'].str.len() == 0).sum()
    print(f"\nEmpty texts after cleaning: {empty_count}")
    
    if empty_count > 0:
        df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)
        print(f"Removed {empty_count} empty texts. New shape: {df.shape}")
    
    return df

# ============================================================================
# STAGE 4: TOKENIZATION
# ============================================================================

class MentalHealthDataset(Dataset):
    """PyTorch Dataset for Mental Health texts"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': self.labels[idx]
        }
        return item


class StreamingMentalHealthDataset(Dataset):
    """
    Memory-efficient dataset that tokenizes each sample on demand.

    This is slower than pre-tokenizing everything once, but significantly reduces
    RAM usage and is usually a better fit for free Google Colab runtimes.
    """

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': self.labels[idx]
        }

def tokenize_data(texts, tokenizer, max_length=512):
    """
    Tokenize texts using RoBERTa tokenizer.

    Note: this tokenizes the full input series in one call and stores all encoded
    tensors in memory. This is fast for moderate sample sizes, but memory-heavy
    for very large datasets.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: TOKENIZATION")
    print("=" * 70)
    
    print(f"\nTokenizing {len(texts)} texts with max_length={max_length}...")
    
    encodings = tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    print(f"✓ Tokenization complete")
    print(f"  input_ids shape: {encodings['input_ids'].shape}")
    print(f"  attention_mask shape: {encodings['attention_mask'].shape}")
    
    return encodings


def tokenize_data_in_chunks(texts, tokenizer, max_length=512, chunk_size=1024):
    """
    Tokenize texts in chunks to reduce peak memory usage during tokenization.

    Note: final encoded tensors are still fully stored in memory. Use
    StreamingMentalHealthDataset for lower steady-state RAM.
    """
    print("\n" + "=" * 70)
    print("STAGE 4: TOKENIZATION (CHUNKED)")
    print("=" * 70)

    total = len(texts)
    print(f"\nTokenizing {total} texts in chunks of {chunk_size} (max_length={max_length})...")

    input_ids_chunks = []
    attention_mask_chunks = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = texts.iloc[start:end].tolist() if hasattr(texts, 'iloc') else texts[start:end]

        enc = tokenizer(
            chunk,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_chunks.append(enc['input_ids'])
        attention_mask_chunks.append(enc['attention_mask'])
        print(f"  Processed rows {start} to {end - 1}")

    encodings = {
        'input_ids': torch.cat(input_ids_chunks, dim=0),
        'attention_mask': torch.cat(attention_mask_chunks, dim=0)
    }

    print("✓ Chunked tokenization complete")
    print(f"  input_ids shape: {encodings['input_ids'].shape}")
    print(f"  attention_mask shape: {encodings['attention_mask'].shape}")

    return encodings

# ============================================================================
# STAGE 5-7: SPLITTING AND DATALOADERS
# ============================================================================

def create_dataloaders(dataset, df_labels, batch_size=32, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train/val/test with stratification
    Create PyTorch DataLoaders
    """
    print("\n" + "=" * 70)
    print("STAGE 5-7: DATA SPLITTING AND DATALOADERS")
    print("=" * 70)
    
    # First split: train / temp (val+test)
    train_idx, temp_idx = train_test_split(
        range(len(dataset)),
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=df_labels
    )
    
    # Second split: val / test
    val_size_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size_ratio),
        random_state=random_state,
        stratify=df_labels.iloc[temp_idx].values
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Print split information
    print(f"\nData split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Total: {len(dataset)} samples")
    
    # Verify class distribution in each split
    print(f"\nClass distribution in splits:")
    for split_name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        split_labels = df_labels.iloc[indices]
        print(f"\n  {split_name}:")
        for label in sorted(split_labels.unique()):
            count = (split_labels == label).sum()
            pct = (count / len(split_labels)) * 100
            print(f"    Class {label}: {count:4d} ({pct:5.1f}%)")
    
    return train_loader, val_loader, test_loader, (train_idx, val_idx, test_idx)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def create_complete_pipeline(
    csv_path,
    batch_size=32,
    sample_size=None,
    tokenization_mode='streamed',
    max_length=512,
    chunk_size=1024,
):
    """
    Execute complete data pipeline
    
    Args:
        csv_path (str): Path to dataset CSV
        batch_size (int): Batch size for DataLoaders
        sample_size (int): Optional - use only first N samples (for testing)
        tokenization_mode (str): 'streamed', 'chunked', or 'full'
        max_length (int): Maximum token length for RoBERTa tokenizer
        chunk_size (int): Chunk size used when tokenization_mode='chunked'
    
    Returns:
        dict: Contains loaders, dataset, tokenizer, and indices
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " MENTAL HEALTH SENTIMENT ANALYSIS - DATA PIPELINE ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Load and explore
    df = load_and_explore_data(csv_path)
    
    # Sample if needed (for quick testing)
    if sample_size and sample_size < len(df):
        print(f"\n⚠️  Using a random sample of {sample_size} rows for testing")

        # Prefer stratified sampling so all classes remain represented.
        min_class_count = df['label'].value_counts().min()
        unique_classes = df['label'].nunique()
        can_stratify = sample_size >= unique_classes and min_class_count >= 2

        if can_stratify:
            _, sampled_df = train_test_split(
                df,
                test_size=sample_size,
                random_state=42,
                stratify=df['label']
            )
            df = sampled_df.reset_index(drop=True)
            print("   Stratified random sampling applied.")
        else:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print("   Random sampling applied (stratify not feasible for this sample size).")
    
    # Preprocess
    df = preprocess_data(df)
    
    # Tokenize / dataset build strategy
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if tokenization_mode == 'streamed':
        print("\nUsing STREAMED tokenization mode (recommended for free Colab RAM).")
        dataset = StreamingMentalHealthDataset(
            texts=df['text_clean'],
            labels=df['label'].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
    elif tokenization_mode == 'chunked':
        encodings = tokenize_data_in_chunks(
            df['text_clean'],
            tokenizer,
            max_length=max_length,
            chunk_size=chunk_size
        )
        dataset = MentalHealthDataset(encodings, df['label'].values)
    elif tokenization_mode == 'full':
        encodings = tokenize_data(df['text_clean'], tokenizer, max_length=max_length)
        dataset = MentalHealthDataset(encodings, df['label'].values)
    else:
        raise ValueError("tokenization_mode must be one of: 'streamed', 'chunked', 'full'")
    
    # Create loaders
    train_loader, val_loader, test_loader, indices = create_dataloaders(
        dataset,
        df['label'],
        batch_size=batch_size
    )
    
    print("\n" + "=" * 70)
    print("✓ DATA PIPELINE COMPLETE")
    print("=" * 70)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'dataset': dataset,
        'df': df,
        'tokenizer': tokenizer,
        'indices': indices,  # (train_idx, val_idx, test_idx)
        'class_names': CURRENT_CLASS_NAMES,
        'status_to_label': CURRENT_STATUS_TO_LABEL,
    }

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    csv_path = "path/to/mental_health_dataset.csv"  # Update this path
    
    try:
        # Create pipeline
        pipeline_data = create_complete_pipeline(
            csv_path,
            batch_size=32,
            sample_size=5000,
            tokenization_mode='streamed',
            max_length=512,
            chunk_size=1024
        )
        
        # Test loading a batch
        print("\n" + "=" * 70)
        print("TESTING: Loading a sample batch")
        print("=" * 70)
        batch = next(iter(pipeline_data['train_loader']))
        print(f"\nBatch contents:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['label'].shape}")
        print(f"  Sample label: {batch['label'][0].item()}")
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {csv_path}")
        print("Please update the csv_path variable with the correct path to your dataset.")
