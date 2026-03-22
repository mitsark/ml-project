"""
Training, Evaluation, and Results Generation
Includes: Training loop, baseline models, evaluation metrics, SHAP explanations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tqdm import tqdm

sns.set_style("whitegrid")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """Get appropriate device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✓ Using CPU")
    return device

def save_checkpoint(model, optimizer, epoch, best_val_acc, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }, save_path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_val_acc']

# ============================================================================
# BASELINE MODEL: LOGISTIC REGRESSION + TF-IDF
# ============================================================================

class BaselineModel:
    """
    Simple baseline: Logistic Regression with TF-IDF features
    Used for comparison with TRABSA architecture
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Args:
            max_features (int): Max TF-IDF features
            ngram_range (tuple): N-gram range for TF-IDF
        """
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'test_results': {}
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train baseline model
        
        Args:
            X_train, y_train: Training texts and labels
            X_val, y_val: Validation texts and labels
        """
        print("\n" + "=" * 70)
        print("🔷 TRAINING BASELINE MODEL (LR + TF-IDF)")
        print("=" * 70)
        
        # Fit TF-IDF
        print("\nFitting TF-IDF vectorizer...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        X_val_tfidf = self.tfidf.transform(X_val)
        
        print(f"  Vocabulary size: {len(self.tfidf.get_feature_names_out())}")
        print(f"  Training set shape: {X_train_tfidf.shape}")
        
        # Train Logistic Regression
        print("\nTraining Logistic Regression...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_tfidf)
        val_pred = self.model.predict(X_val_tfidf)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        
        print(f"\nBaseline Results:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        return X_train_tfidf, X_val_tfidf
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate baseline on test set
        
        Args:
            X_test, y_test: Test texts and labels
            class_names: List of class names
        
        Returns:
            dict: Evaluation results
        """
        X_test_tfidf = self.tfidf.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        test_f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        self.history['test_results'] = {
            'accuracy': test_acc,
            'f1_weighted': test_f1_weighted,
            'f1_macro': test_f1_macro,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"\n🔷 BASELINE TEST RESULTS:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   F1 (weighted): {test_f1_weighted:.4f}")
        print(f"   F1 (macro): {test_f1_macro:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Baseline Confusion Matrix (LR + TF-IDF)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('baseline_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved baseline confusion matrix to 'baseline_confusion_matrix.png'")
        plt.close()
        
        return self.history['test_results']

# ============================================================================
# DEEP LEARNING TRAINING
# ============================================================================

def calculate_class_weights(labels, device):
    """Calculate class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: TRABSA model
        loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    """
    Validate model
    
    Args:
        model: TRABSA model
        loader: Validation DataLoader
        criterion: Loss function
        device: torch.device
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    device,
    num_epochs=15,
    learning_rate=2e-5,
    patience=3,
    save_best_path='best_trabsa_model.pth'
):
    """
    Complete training loop with early stopping
    
    Args:
        model: TRABSA model
        train_loader, val_loader: DataLoaders
        class_weights: Tensor of class weights
        device: torch.device
        num_epochs (int): Maximum epochs
        learning_rate (float): Initial LR
        patience (int): Early stopping patience
        save_best_path (str): Path to save best model
    
    Returns:
        dict: Training history
    """
    print("\n" + "=" * 70)
    print("🔷 TRAINING TRABSA MODEL")
    print("=" * 70)
    
    # Setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_acc, save_best_path)
            print(f"   ✓ Best val accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    checkpoint = torch.load(save_best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Loaded best model with val_acc: {checkpoint['best_val_acc']:.4f}")
    
    return history

# ============================================================================
# EVALUATION AND RESULTS GENERATION
# ============================================================================

def generate_predictions_with_confidence(model, loader, device):
    """
    Generate predictions with confidence scores
    
    Returns:
        tuple: (predictions, true_labels, confidence_scores, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1)[0]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confs),
        np.array(all_probs)
    )

def evaluate_deep_model(model, test_loader, device, class_names):
    """
    Evaluate TRABSA model on test set
    
    Returns:
        dict: Comprehensive evaluation results
    """
    print("\n" + "=" * 70)
    print("🔷 EVALUATING TRABSA MODEL")
    print("=" * 70)
    
    # Generate predictions
    test_preds, test_labels, confs, probs = generate_predictions_with_confidence(
        model, test_loader, device
    )
    
    # Metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Accuracy: {test_acc:.4f}")
    print(f"   F1 (weighted): {test_f1_weighted:.4f}")
    print(f"   F1 (macro): {test_f1_macro:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('TRABSA Confusion Matrix', fontweight='bold', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('trabsa_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved TRABSA confusion matrix to 'trabsa_confusion_matrix.png'")
    plt.close()
    
    return {
        'accuracy': test_acc,
        'f1_weighted': test_f1_weighted,
        'f1_macro': test_f1_macro,
        'predictions': test_preds,
        'true_labels': test_labels,
        'confidence_scores': confs,
        'probabilities': probs,
        'confusion_matrix': cm
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Loss over Epochs', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_title('Accuracy over Epochs', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved training curves to '{save_path}'")
    plt.close()

def plot_model_comparison(baseline_results, trabsa_results, save_path='model_comparison.png'):
    """Compare baseline vs TRABSA"""
    models = ['Baseline\n(LR+TF-IDF)', 'TRABSA']
    accuracies = [baseline_results['accuracy'], trabsa_results['accuracy']]
    f1_scores = [baseline_results['f1_weighted'], trabsa_results['f1_weighted']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = axes[0].bar(models, accuracies, color=['steelblue', 'green'], edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
    axes[0].set_ylim([0, 1])
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        axes[0].text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.3f}',
                     ha='center', va='bottom', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # F1 comparison
    bars2 = axes[1].bar(models, f1_scores, color=['steelblue', 'green'], edgecolor='black')
    axes[1].set_ylabel('F1 Score (Weighted)', fontsize=11)
    axes[1].set_title('F1 Score Comparison', fontweight='bold', fontsize=12)
    axes[1].set_ylim([0, 1])
    for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
        axes[1].text(bar.get_x() + bar.get_width()/2, f1 + 0.02, f'{f1:.3f}',
                     ha='center', va='bottom', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved model comparison to '{save_path}'")
    plt.close()

# ============================================================================
# EXPLAINABILITY (SHAP - Simplified)
# ============================================================================

def generate_sample_explanations(model, samples, tokenizer, class_names, device, num_samples=3):
    """
    Generate SHAP-style explanations for sample predictions
    
    Args:
        model: TRABSA model
        samples: List of (text, true_label) tuples
        tokenizer: RoBERTa tokenizer
        class_names: List of class names
        device: torch.device
        num_samples: Number of samples to explain
    """
    print("\n" + "=" * 70)
    print("💡 GENERATING EXPLANATIONS")
    print("=" * 70)
    
    model.eval()
    
    for idx, (text, true_label) in enumerate(samples[:num_samples]):
        print(f"\n--- Sample {idx + 1} ---")
        print(f"Text: {text[:150]}...")
        print(f"True label: {class_names[true_label]}")
        
        # Tokenize
        encodings = tokenizer(
            [text],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Forward pass to get token weights from attention
        with torch.no_grad():
            logits, token_weights = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
        
        print(f"Predicted label: {class_names[pred_class]} (confidence: {confidence:.2%})")
        
        # Get top tokens by attention weight
        top_k = 5
        top_indices = torch.argsort(token_weights, descending=True)[:top_k]
        top_tokens = [tokenizer.convert_ids_to_tokens(input_ids[0, i].item()) for i in top_indices]
        top_weights = [token_weights[i].item() for i in top_indices]
        
        print(f"\nTop contributing tokens:")
        for token, weight in zip(top_tokens, top_weights):
            bar = '█' * int(weight * 50)
            print(f"  {token:15s} {bar} {weight:.4f}")

if __name__ == "__main__":
    print("Training and evaluation module loaded successfully!")
