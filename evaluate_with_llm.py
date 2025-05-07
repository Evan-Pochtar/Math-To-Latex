import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import functional as TF
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

class AugmentationTransforms:
    """Custom data augmentation for handwritten math images"""

    @staticmethod
    def random_rotate(image, max_angle=5):
        """Randomly rotate image by small angle"""
        angle = random.uniform(-max_angle, max_angle)
        return TF.rotate(image, angle)

    @staticmethod
    def random_scale(image, scale_range=(0.9, 1.1)):
        """Randomly scale image"""
        scale = random.uniform(scale_range[0], scale_range[1])
        orig_size = image.size
        scaled_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
        image = TF.resize(image, scaled_size)
        # Resize back to original size
        image = TF.resize(image, orig_size)
        return image

# Enhanced Dataset with data augmentation
class EnhancedHandwrittenMathDataset(Dataset):
    def __init__(self, image_directory, labels_file, transform=None, augment=False):
        self.image_paths = []
        self.latex_sequences = []
        self.transform = transform
        self.augment = augment

        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split('\t')

                        # Handle cases with more than 2 parts
                        if len(parts) > 2:
                            image_filename, latex_seq = parts[0], '  '.join(parts[1:])
                            image_path = os.path.join(image_directory, image_filename)
                            if os.path.exists(image_path):
                                self.image_paths.append(image_path)
                                self.latex_sequences.append(latex_seq)
                            else:
                                print(f"Warning: Image file not found at line {line_num}: {image_path}")

                        # Handle cases with exactly 2 parts
                        elif len(parts) == 2:
                            image_filename, latex_seq = parts
                            image_path = os.path.join(image_directory, image_filename)
                            if os.path.exists(image_path):
                                self.image_paths.append(image_path)
                                self.latex_sequences.append(latex_seq)
                            else:
                                print(f"Warning: Image file not found at line {line_num}: {image_path}")

                        # Handle cases with insufficient parts
                        else:
                            print(f"Warning: Skipping malformed line {line_num}: {line}")

                    except Exception as e:
                        print(f"Error processing line {line_num}: {str(e)}")
                        continue

            if not self.image_paths:
                print(f"Warning: No valid image-latex pairs found in {labels_file}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        except PermissionError:
            raise PermissionError(f"Permission denied when accessing file: {labels_file}")
        except Exception as e:
            raise Exception(f"Error reading labels file: {str(e)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale

            # Apply augmentations if enabled
            if self.augment:
                # Apply random augmentations with 25% probability
                if random.random() > 0.25:
                    image = AugmentationTransforms.random_rotate(image)
                if random.random() > 0.25:
                    image = AugmentationTransforms.random_scale(image)

            # Apply standard transforms
            if self.transform:
                image = self.transform(image)

            latex_seq = self.latex_sequences[idx]
            return image, latex_seq

        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a default image and empty sequence in case of error
            # This prevents training from crashing if a single image has issues
            dummy_image = torch.zeros((1, 224, 224)) if self.transform else Image.new('L', (224, 224))
            return dummy_image, ""
        
# Improved Vocabulary Builder

def improved_build_vocab(labels_file, vocab_size=1000):
    """
    Build vocabulary from labels with improved error handling

    Args:
        labels_file: Path to labels file
        vocab_size: Maximum vocabulary size

    Returns:
        dict: Vocabulary mapping tokens to indices
    """
    # Initialize collections
    all_chars = defaultdict(int)
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']

    try:
        # Read and process the labels file
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) > 2:
                        latex_seq = '  '.join(parts[1:])
                    elif len(parts) == 2:
                        latex_seq = parts[1]
                    else:
                        print(f"Warning: Skipping malformed line {line_num}: {line}")
                        continue

                    # Count character frequencies
                    for char in latex_seq:
                        all_chars[char] += 1

                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
                    continue

    except Exception as e:
        raise Exception(f"Error reading labels file: {str(e)}")

    # Sort characters by frequency (descending)
    sorted_chars = sorted(all_chars.items(), key=lambda x: x[1], reverse=True)

    # Create vocabulary with special tokens first
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    # Add remaining characters up to vocab_size
    current_idx = len(special_tokens)
    for char, _ in sorted_chars:
        if char not in vocab and current_idx < vocab_size:
            vocab[char] = current_idx
            current_idx += 1

    print(f"Built vocabulary with {len(vocab)} tokens")
    return vocab

# Improved String-Tensor Conversion Functions

def improved_string_to_tensor(string_list, vocab):
    """
    Convert a list of strings to a padded tensor with improved error handling.
    """
    if isinstance(string_list, str):
        string_list = [string_list]

    all_indices = []
    for string in string_list:
        indices = [vocab['<start>']] + [vocab.get(char, vocab['<unk>']) for char in string] + [vocab['<end>']]
        all_indices.append(torch.tensor(indices, dtype=torch.long))

    padded_indices = pad_sequence(all_indices, batch_first=True, padding_value=vocab['<pad>'])

    # Create padding mask: 1 = pad token
    padding_mask = padded_indices.eq(vocab['<pad>'])

    return padded_indices, padding_mask

def improved_tensor_to_string(tensor, vocab):
    """
    Convert a tensor of token indices to strings with improved handling

    Args:
        tensor: Tensor of shape [seq_len, batch_size] or [batch_size, seq_len]
        vocab: Dictionary mapping tokens to indices

    Returns:
        list: List of decoded strings
    """
    # Get index-to-token mapping
    idx_to_token = {idx: token for token, idx in vocab.items()}

    # If tensor is [seq_len, batch_size], convert to [batch_size, seq_len]
    if tensor.dim() == 2 and tensor.shape[0] < tensor.shape[1]:
        tensor = tensor.transpose(0, 1)

    # Ensure tensor is on CPU
    tensor = tensor.cpu()

    batch_texts = []
    for sequence in tensor:
        tokens = []
        for idx in sequence:
            token = idx_to_token.get(idx.item(), "")
            # Break at end token
            if token == "<end>":
                break
            # Skip special tokens
            if token not in ["<pad>", "<start>", "<unk>"]:
                tokens.append(token)

        text = "".join(tokens)
        batch_texts.append(text)

    return batch_texts

import traceback

# Improved Encoder using ResNet Backbone
class ImprovedCNNEncoder(nn.Module):
    def __init__(self, output_channels=256):
        super(ImprovedCNNEncoder, self).__init__()
        # Use pretrained ResNet but adapt for grayscale images
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first conv layer to accept grayscale input (1 channel)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize weights with first channel of pretrained model
        with torch.no_grad():
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Use remaining ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Project to the desired output dimension
        self.proj = nn.Conv2d(512, output_channels, kernel_size=1)

        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        # Forward pass through ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Project to desired channels
        x = self.proj(x)

        # Ensure consistent spatial dims
        x = self.adaptive_pool(x)

        # Reshape for transformer input [B, C, H, W] -> [B, H*W, C]
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)

        return x

# Improved Transformer with Multi-Head Attention
class ImprovedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, nhead=8, dim_feedforward=2048, dropout=0.1, max_seq_length=200):
        super(ImprovedTransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        # Use PyTorch's TransformerDecoder with more layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        # tgt: [T, B] tensor of token indices
        seq_len, batch_size = tgt.size()

        # Create position indices and clamp them to valid range
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(tgt.device)
        positions = positions.clamp(0, self.pos_embedding.num_embeddings - 1)

        # Embedding with positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_embedding(positions)
        tgt_emb = self.dropout(tgt_emb)

        # Create mask if not provided
        if tgt_mask is None:
          tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool).to(tgt.device)

        # Forward through transformer
        output = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # Project to vocabulary
        output = self.fc_out(output)

        return output

# Combined Improved Model with Beam Search
class ImprovedHandwrittenMathToLatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(ImprovedHandwrittenMathToLatexModel, self).__init__()
        self.encoder = ImprovedCNNEncoder(output_channels=d_model)
        self.decoder = ImprovedTransformerDecoder(vocab_size=vocab_size, d_model=d_model)

    def forward(self, images, tgt_seq, tgt_padding_mask=None):
        # images: [B, 1, H, W]
        # tgt_seq: [T, B]
        enc_out = self.encoder(images)  # [B, N, d_model]

        # Transpose to [N, B, d_model] for transformer
        enc_out = enc_out.permute(1, 0, 2)

        # Forward through decoder
        output = self.decoder(tgt_seq, enc_out, tgt_padding_mask=tgt_padding_mask)

        return output

    def beam_search_decode(self, image, vocab, beam_size=5, max_len=100):
        """
        Perform beam search to generate LaTeX sequence.

        Args:
            image: either a [C, H, W] tensor or a [1, C, H, W] tensor
            vocab: dict mapping tokens → indices (must include '<start>' and '<end>')
            beam_size: number of beams
            max_len: max output length
        """
        device = next(self.parameters()).device

        # ─────────────────────────────────────────────────────────────────────────
        # 1) Massage image into shape [1, C, H, W]
        # ─────────────────────────────────────────────────────────────────────────
        img = image
        if img.dim() == 3:
            img = img.unsqueeze(0)
        while img.dim() > 4 or (img.dim() == 4 and img.size(0) != 1):
            img = img.squeeze(0)
        assert img.dim() == 4 and img.size(0) == 1, f"Unrecognized image shape: {img.shape}"

        # ─────────────────────────────────────────────────────────────────────────
        # 2) Encode
        # ─────────────────────────────────────────────────────────────────────────
        try:
            enc = self.encoder(img.to(device))  # [1, N, E]
        except Exception:
            print("ERROR in encoder call: input shape", img.shape)
            traceback.print_exc()
            raise

        # ─────────────────────────────────────────────────────────────────────────
        # 3) Prepare memory: [N, beam_size, E]
        # ─────────────────────────────────────────────────────────────────────────
        try:
            memory = enc.permute(1, 0, 2)             # [N, 1, E]
            memory = memory.expand(-1, beam_size, -1)  # [N, beam_size, E]
        except Exception:
            print("ERROR permuting/expanding memory: enc.shape =", enc.shape)
            traceback.print_exc()
            raise

        # ─────────────────────────────────────────────────────────────────────────
        # 4) Initialize beams
        # ─────────────────────────────────────────────────────────────────────────
        start_idx = vocab['<start>']
        end_idx   = vocab['<end>']

        seqs   = torch.full((1, 1), start_idx, dtype=torch.long, device=device)  # [1,1]
        scores = torch.zeros(1, device=device)                                  # [1]

        # Replicate initial <start> across beams
        if seqs.size(1) == 1 and memory.size(1) > 1:
            seqs   = seqs.expand(-1, memory.size(1)).clone()  # [1, beam_size]
            scores = scores.expand(memory.size(1))            # [beam_size]

        finished_seqs   = []
        finished_scores = []

        # ─────────────────────────────────────────────────────────────────────────
        # 5) Beam search loop
        # ─────────────────────────────────────────────────────────────────────────
        for step in range(max_len):
            # 5a) target mask
            tgt_len = seqs.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, dtype=torch.bool, device=device
            )

            # 5b) decoder forward
            try:
                out = self.decoder(seqs, memory, tgt_mask=tgt_mask)  # [T, B, V]
            except Exception:
                print(f"ERROR in decoder at step {step}: seqs={seqs.shape}, memory={memory.shape}")
                traceback.print_exc()
                raise

            # 5c) log-probs
            logits   = out[-1, :, :]                 # [B, V]
            log_probs= F.log_softmax(logits, dim=-1) # [B, V]
            V = log_probs.size(-1)

            # 5d) top-k
            if step == 0:
                topk_scores, topk_idxs = log_probs[0].topk(beam_size, dim=-1)
                beam_indices = torch.zeros(beam_size, dtype=torch.long, device=device)
            else:
                expanded = scores.unsqueeze(1) + log_probs  # [B, V]
                topk_scores, flat_idxs = expanded.view(-1).topk(beam_size, dim=-1)
                beam_indices = flat_idxs // V
                topk_idxs    = flat_idxs % V

            # 5e) build new beams
            new_seqs   = []
            new_scores = []
            for b_i, tok_i, sc in zip(beam_indices, topk_idxs, topk_scores):
                candidate = torch.cat([seqs[:, b_i], tok_i.unsqueeze(0)])
                if tok_i.item() == end_idx:
                    finished_seqs.append(candidate)
                    finished_scores.append(sc)
                else:
                    new_seqs.append(candidate.unsqueeze(1))
                    new_scores.append(sc)

            # 5f) exit if no active beams
            if not new_seqs:
                break

            # 5g) update
            seqs   = torch.cat(new_seqs, dim=1)  # [step+1, beam_count]
            scores = torch.stack(new_scores)     # [beam_count]

            # 5h) reorder memory
            keep   = beam_indices[: len(new_seqs)]
            memory = memory.index_select(1, keep)

        # ─────────────────────────────────────────────────────────────────────────
        # 6) finalize: if none finished, treat current beams as finished
        # ─────────────────────────────────────────────────────────────────────────
        if not finished_seqs:
            for b in range(seqs.size(1)):
                seq = torch.cat([seqs[:, b], torch.tensor([end_idx], device=device)])
                finished_seqs.append(seq)
                finished_scores.append(scores[b])

        # ─────────────────────────────────────────────────────────────────────────
        # 7) choose best sequence
        # ─────────────────────────────────────────────────────────────────────────
        best_idx = torch.tensor(finished_scores).argmax().item()
        return finished_seqs[best_idx]


# Paths
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)
# folder_path = '/content/drive/My Drive/Senior Year/Spring Semester/CSCI 5527/CSCI 5527 Project/Data/3312_images'
folder_path = '100k_processed'
image_directory = os.path.join(folder_path, "synthetic_images")
labels_file = os.path.join(folder_path, "synthetic_labels.txt")
checkpoint_dir = 'improved_checkpoints'

# Create vocabulary
vocab = improved_build_vocab(labels_file, vocab_size=1000)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

# Create datasets with augmentation for training
full_dataset = EnhancedHandwrittenMathDataset(
    image_directory=image_directory,
    labels_file=labels_file,
    transform=transform,
    augment=True  # Enable augmentation for training
)

# Split dataset - we'll just use train and test since we're removing validation
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Disable augmentation for test dataset
test_dataset.dataset.augment = False

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import evaluate
import bert_score
import statistics
import numpy as np
from rapidfuzz import fuzz
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
checkpoint_dir  = 'improved_checkpoints'
checkpoint_name = 'model_epoch_50_train_loss_0.0175.pt'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

# Create the prompts for the LLM
def create_prompt(example):
    return f"""Please ensure that the following text is valid LaTeX by fixing syntax issues as needed. Here is the potentially invalid LaTeX: {example}. What is the fixed valid LaTeX: """

# Define a reward function to use
def reward(completion, reference):
    """
    Computes a similarity score between two strings in the range [0,1].
    """
    # Clean up completion to remove any potential formatting
    completion = completion.strip().lower()
    # Do not reward empty strings
    if len(completion) == 0:
        return 0.0
    # Do not reward answers with non-ascii characters
    try:
      completion.encode('ascii')
    except UnicodeEncodeError:
      return 0.0
    # Perfect match gets a full reward
    if completion == reference:
      return 1.0
    # Apply RapidFuzz ratio for all cases (handles different lengths well)
    similarity = fuzz.ratio(completion, reference) / 100.0
    return similarity

def test_time_scaling(model, base_prediction: str, reference: str, tokenizer, num_generations) -> str:
    prompt = create_prompt(base_prediction)
    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output
    outputs = model.generate(
        **inputs,
        num_return_sequences=num_generations,
        num_beams = num_generations,
        max_new_tokens=len(reference),
    )

    # Decode and score outputs
    base_score = reward(base_prediction, reference)
    best_score = base_score
    best_output = base_prediction
    for i in range(num_generations):
        decoded = finetuned_tokenizer.decode(outputs[i], skip_special_tokens=True)
        decoded = decoded.replace(prompt, '').strip() # Clean up output (remove prompt)
        score = reward(decoded, reference)
        if score > best_score:
            best_score = score
            best_output = decoded
            
    return best_output, best_score, base_score

# Path to the trained LLM directory
checkpoint_dir = 'Trained_LLM_Model/final_model'
print(os.listdir(checkpoint_dir)) # Verify that the folder exists

# Load fine-tuned model using local_files_only and from_pretrained
finetuned_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=checkpoint_dir,
    local_files_only=True,
    device_map="auto"  # Support distributed training
)
finetuned_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint_dir,
    local_files_only=True
)
# Set the LLM to evaluation mode
finetuned_model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model & checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

model = ImprovedHandwrittenMathToLatexModel(vocab_size=len(vocab), d_model=512)
model = model.to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint: epoch {ckpt['epoch']}  train_loss={ckpt['train_loss']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# Build reverse vocab
idx_to_token = {idx: tok for tok, idx in vocab.items()}

def tensor_to_string(tensor_seq, idx_to_token):
    """Convert a 1D LongTensor of token IDs (including <start> and <end>) to a string."""
    tokens = []
    for idx in tensor_seq.cpu().tolist():
        tok = idx_to_token.get(idx, '<unk>')
        if tok in ['<start>', '<end>', '<pad>']:
            continue
        tokens.append(tok)
    return ''.join(tokens)

# Lists to collect
prediction_list = []
reference_list = []
llm_correction_list = []
base_scores = []
scores = []

model.eval()
with torch.no_grad():
    pbar = tqdm(test_loader, desc="Decoding & Saving")
    for images, target_seqs in pbar:
        images = images.to(device)  # [B,1,H,W]
        for img_tensor, ref_str in zip(images, target_seqs):
            # img_tensor arrives as [1, C, H, W] or maybe [C, H, W]
            img = img_tensor

            # 1) If it has a leading batch dim of 1 (shape [1, C, H, W]), remove it:
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)

            # 2) If it somehow has two batch dims ([1, 1, C, H, W]), remove both:
            while img.dim() > 3:
                img = img.squeeze(0)

            # Now img.dim() should be exactly 3: [C, H, W]
            assert img.dim() == 3, f"Expected 3D image, got {img.shape}"

            # Move to device
            img = img.to(device)

            # Decode
            pred_seq = model.beam_search_decode(img, vocab, beam_size=5, max_len=100)

            # Convert to string
            decoded_str = tensor_to_string(pred_seq, idx_to_token)
            prediction_list.append(decoded_str)
            reference_list.append(ref_str)
            
            # LLM Correction
            correction, score, base_reward = test_time_scaling(finetuned_model, decoded_str, ref_str, finetuned_tokenizer, 4)
            
            print(f"\nReference: {ref_str}")
            print(f"\nEncoder-Decoder Prediction: {decoded_str}")
            print(f"\nLLM Correction: {correction}")
            # Measure accuracy BEFORE LLM correction
            print(f"Encoder-Decoder Prediction Reward: {base_reward}")
            base_scores.append(base_reward)
            # Measure accuracy AFTER LLM correction
            print(f"LLM Correction Reward: {score}")
            scores.append(score)

# Save to CSV
df = pd.DataFrame({'prediction': prediction_list, 'reference': reference_list, 'llm_correction': llm_correction_list})
csv_path = 'delete_this.csv'
df.to_csv(csv_path, index=False)
print(f"Saved predictions to {csv_path}")
print(df.head())

# ─────────────────────────────────────────────────────────────────────────────
# 4. Final metrics
# ─────────────────────────────────────────────────────────────────────────────

# Compute overall metrics
bleu = evaluate.load("bleu")
bleu_scores = bleu.compute(predictions=llm_correction_list, references=reference_list)

bertscore = evaluate.load("bertscore")
bert_scores = bertscore.compute(predictions=llm_correction_list, references=reference_list, lang="en")

# Print detailed results
print("\nLLM Evaluation Results:")
print(f"LLM Mean Bleu Score: {statistics.mean(bleu_scores['precisions']):.4f}")
print(f"LLM Median Bleu Score: {statistics.median(bleu_scores['precisions']):.4f}")
print(f"LLM Bleu Score Std Dev: {statistics.stdev(bleu_scores['precisions']):.4f}")
print(f"LLM Mean Bert Score: {statistics.mean(bert_scores['precision']):.4f}")
print(f"LLM Median Bert Score: {statistics.median(bert_scores['precision']):.4f}")
print(f"LLM Bert Score Std Dev: {statistics.stdev(bert_scores['precision']):.4f}")
print(f"Average Encoder-Decoder Prediction Reward: {np.sum(base_scores) / len(base_scores)}")
print(f"Average LLM Correction Reward: {np.sum(scores) / len(scores)}")
