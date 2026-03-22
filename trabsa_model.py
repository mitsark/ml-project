"""
TRABSA Model Architecture Implementation
Stage 1: RoBERTa Feature Extraction
Stage 2: Attention Layer
Stage 3: Bidirectional LSTM
Stage 4: Classification Head
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
import numpy as np

# ============================================================================
# STAGE 1: RoBERTa FEATURE EXTRACTOR
# ============================================================================

class RobertaFeatureExtractor(nn.Module):
    """
    RoBERTa-based feature extraction with selective layer freezing
    to prevent overfitting on small datasets
    """
    
    def __init__(self, model_name='roberta-base', freeze_layers=10, output_hidden_states=False):
        """
        Args:
            model_name (str): Pre-trained RoBERTa model name
            freeze_layers (int): Number of early layers to freeze
            output_hidden_states (bool): Return all hidden states if True
        """
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name, output_hidden_states=output_hidden_states)
        
        # Print model information
        total_layers = len(self.roberta.encoder.layer)
        print(f"\n📊 RoBERTa Extractor:")
        print(f"   Total layers: {total_layers}")
        print(f"   Freezing first {freeze_layers} layers")
        
        # Freeze early layers to prevent overfitting
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.roberta.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.roberta.parameters())
        print(f"   Trainable params: {trainable_params:,} / {total_params:,}")
    
    def forward(self, input_ids, attention_mask):
        """
        Extract contextual embeddings from RoBERTa
        
        Args:
            input_ids (torch.Tensor): Token IDs (batch_size, seq_len)
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len)
        
        Returns:
            last_hidden_state (torch.Tensor): (batch_size, seq_len, 768)
            pooled_output (torch.Tensor): (batch_size, 768) - [CLS] representation
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.last_hidden_state, outputs.pooler_output

# ============================================================================
# STAGE 2: CUSTOM ATTENTION LAYER
# ============================================================================

class AttentionLayer(nn.Module):
    """
    Query-Key-Value based attention mechanism
    Computes token importance weights
    """
    
    def __init__(self, input_dim=768, num_heads=8):
        """
        Args:
            input_dim (int): Dimension of input embeddings (768 for RoBERTa)
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim ** -0.5
        
        # Output projection
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        
        print(f"\n🎯 Attention Layer:")
        print(f"   Input dim: {input_dim}")
        print(f"   Num heads: {num_heads}")
        print(f"   Head dim: {self.head_dim}")
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Multi-head attention mechanism
        
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, 768)
            attention_mask (torch.Tensor): (batch_size, seq_len) - 1 for real, 0 for padding
        
        Returns:
            context (torch.Tensor): (batch_size, seq_len, 768) - attended hidden states
            token_weights (torch.Tensor): (seq_len,) - average importance per token
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        Q = self.query(hidden_states)  # (batch, seq_len, 768)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Each: (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (batch, num_heads, seq_len, seq_len)
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) → expand for broadcasting
            attention_mask_expanded = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(attention_mask_expanded == 0, -1e9)
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        context = torch.matmul(attn_weights, V)
        # (batch, num_heads, seq_len, head_dim)
        
        # Reshape back to single head
        context = context.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        context = context.view(batch_size, seq_len, self.input_dim)  # (batch, seq_len, 768)
        
        # Output projection
        context = self.fc_out(context)
        context = self.dropout(context)
        
        # Compute token importance weights (average across heads and batch)
        # Average attention weights across batch and heads
        token_weights = attn_weights.mean(dim=(0, 1))  # (seq_len,)
        
        return context, token_weights

# ============================================================================
# STAGE 3: BIDIRECTIONAL LSTM LAYER
# ============================================================================

class BiLSTMLayer(nn.Module):
    """
    Bidirectional LSTM for sequential context capture
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=1, dropout=0.3):
        """
        Args:
            input_dim (int): Input feature dimension (768 from attention output)
            hidden_dim (int): Hidden state dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        output_dim = hidden_dim * 2  # Bidirectional
        
        print(f"\n🔄 BiLSTM Layer:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Bidirectional output: {output_dim}")
        print(f"   Num layers: {num_layers}")
    
    def forward(self, x, attention_mask=None):
        """
        Process sequence through BiLSTM
        
        Args:
            x (torch.Tensor): (batch_size, seq_len, 768) - attention layer output
            attention_mask (torch.Tensor): (batch_size, seq_len)
        
        Returns:
            output (torch.Tensor): (batch_size, seq_len, 512)
            final_state (torch.Tensor): (batch_size, 512)
        """
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(x)
        # output: (batch, seq_len, hidden_dim * 2)
        # hidden: (num_layers * 2, batch, hidden_dim) - last hidden state from each direction
        
        output = self.dropout(output)
        
        # Take final hidden state from both directions
        # hidden[-2] = forward direction final state
        # hidden[-1] = backward direction final state
        final_state = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # final_state: (batch, hidden_dim * 2)
        
        return output, final_state

# ============================================================================
# STAGE 4: CLASSIFICATION HEAD
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Fully connected layers for multiclass classification
    """
    
    def __init__(self, input_dim=512, num_classes=7, dropout=0.3):
        """
        Args:
            input_dim (int): Input dimension (512 from BiLSTM)
            num_classes (int): Number of classes
            dropout (float): Dropout rate for regularization
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # First fully connected layer with ReLU activation
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Output layer
        self.fc_out = nn.Linear(128, num_classes)
        
        print(f"\n📊 Classification Head:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden layers: 256 → 128")
        print(f"   Output classes: {num_classes}")
    
    def forward(self, x):
        """
        Classify based on BiLSTM final state
        
        Args:
            x (torch.Tensor): (batch_size, 512) - final BiLSTM state
        
        Returns:
            logits (torch.Tensor): (batch_size, num_classes)
        """
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        logits = self.fc_out(x)
        
        return logits

# ============================================================================
# COMPLETE TRABSA MODEL
# ============================================================================

class TRABSA(nn.Module):
    """
    Full TRABSA (Transformer + Attention + BiLSTM) Architecture
    
    Pipeline:
    Input Text → RoBERTa (Stage 1)
             → Attention (Stage 2)
             → BiLSTM (Stage 3)
             → Classification Head (Stage 4)
             → Class Predictions
    """
    
    def __init__(
        self,
        num_classes=7,
        freeze_roberta_layers=10,
        hidden_dim=256,
        dropout=0.3,
        num_lstm_layers=1,
        num_attention_heads=8
    ):
        """
        Args:
            num_classes (int): Number of mental health classes
            freeze_roberta_layers (int): Number of RoBERTa layers to freeze
            hidden_dim (int): Hidden dimension for BiLSTM
            dropout (float): Dropout rate
            num_lstm_layers (int): Number of LSTM layers (1-2)
            num_attention_heads (int): Number of attention heads (8 or 12)
        """
        super().__init__()
        
        print("\n" + "=" * 70)
        print("🔨 BUILDING TRABSA MODEL ARCHITECTURE")
        print("=" * 70)
        
        # Stage 1: RoBERTa Feature Extraction
        self.roberta_extractor = RobertaFeatureExtractor(
            model_name='roberta-base',
            freeze_layers=freeze_roberta_layers
        )
        
        # Stage 2: Attention Layer
        self.attention_layer = AttentionLayer(
            input_dim=768,
            num_heads=num_attention_heads
        )
        
        # Stage 3: Bidirectional LSTM
        self.bilstm_layer = BiLSTMLayer(
            input_dim=768,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )
        
        # Stage 4: Classification Head
        self.classifier = ClassificationHead(
            input_dim=hidden_dim * 2,  # Bidirectional
            num_classes=num_classes,
            dropout=dropout
        )
        
        print("\n✓ Model architecture complete")
        print("=" * 70)
    
    def forward(self, input_ids, attention_mask):
        """
        Complete forward pass through TRABSA architecture
        
        Args:
            input_ids (torch.Tensor): (batch_size, 512) - RoBERTa token IDs
            attention_mask (torch.Tensor): (batch_size, 512) - RoBERTa attention mask
        
        Returns:
            logits (torch.Tensor): (batch_size, num_classes) - class logits
            token_weights (torch.Tensor): (seq_len,) - token importance from attention
        """
        # Stage 1: RoBERTa Feature Extraction
        hidden_states, pooled_output = self.roberta_extractor(input_ids, attention_mask)
        # hidden_states: (batch, 512, 768)
        
        # Stage 2: Attention Layer
        context, token_weights = self.attention_layer(hidden_states, attention_mask)
        # context: (batch, 512, 768)
        # token_weights: (512,)
        
        # Stage 3: BiLSTM Layer
        lstm_output, final_state = self.bilstm_layer(context, attention_mask)
        # lstm_output: (batch, 512, 512)
        # final_state: (batch, 512)
        
        # Stage 4: Classification Head
        logits = self.classifier(final_state)
        # logits: (batch, num_classes)
        
        return logits, token_weights
    
    def get_model_size(self):
        """Print model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 Model Size:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Parameter size: {total_params * 4 / 1024 / 1024:.2f} MB")

# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    import torch
    
    print("\n" + "=" * 70)
    print("🧪 TESTING TRABSA MODEL")
    print("=" * 70)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 Using device: {device}")
    
    model = TRABSA(
        num_classes=7,
        freeze_roberta_layers=10,
        hidden_dim=256,
        dropout=0.3,
        num_lstm_layers=1,
        num_attention_heads=8
    )
    model.to(device)
    
    # Print model size
    model.get_model_size()
    
    # Create dummy input for testing
    batch_size = 2
    seq_len = 512
    
    dummy_input_ids = torch.randint(0, 50265, (batch_size, seq_len)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    print(f"\n📝 Test Input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    
    # Forward pass
    print(f"\n⚙️  Running forward pass...")
    with torch.no_grad():
        logits, token_weights = model(dummy_input_ids, dummy_attention_mask)
    
    # Output shapes
    print(f"\n📊 Output Shapes:")
    print(f"   Logits: {logits.shape}")
    print(f"   Token weights: {token_weights.shape}")
    
    # Predictions
    predictions = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)
    confidence_scores = torch.max(predictions, dim=1)[0]
    
    print(f"\n🎯 Predictions:")
    for i in range(batch_size):
        pred_class = predicted_classes[i].item()
        confidence = confidence_scores[i].item()
        print(f"   Sample {i+1}: Class {pred_class} (confidence: {confidence:.2%})")
    
    print("\n✓ Test complete - model is ready for training!")
    print("=" * 70)
