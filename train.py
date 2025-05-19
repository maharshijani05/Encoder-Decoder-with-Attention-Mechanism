import torch
import sys
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TranslationDataset, collate_fn
from model import Encoder, Decoder, Seq2Seq, Attention
import os

# -------------------- Hyperparameters --------------------
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.5
BATCH_SIZE = 32
N_EPOCHS = 15
CLIP = 5
CHECKPOINT_DIR = "checkpoints"
TOKENIZED_FILE = r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt"

# -------------------- GPU Check --------------------
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("🚀 Using GPU for training.")
else:
    print("❌ GPU not available. Please run this script on a machine with CUDA-enabled GPU.")
    sys.exit(1)

# -------------------- Load Dataset --------------------
train_dataset = TranslationDataset(TOKENIZED_FILE, min_freq=2, max_len=50)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -------------------- Build Model --------------------
SRC_VOCAB_SIZE = len(train_dataset.src_vocab)
TRG_VOCAB_SIZE = len(train_dataset.trg_vocab)

attention = Attention(HID_DIM)
encoder = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attention)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.trg_vocab.word2idx["<pad>"])

# -------------------- Resume Support --------------------
RESUME = True
RESUME_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_5.pt")  # Update to last saved

start_epoch = 0
if RESUME and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"🔄 Resumed from epoch {start_epoch}")

# -------------------- Training Function --------------------
def train(model, loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(loader, desc="🔁 Training", leave=False)

    for src, trg in progress_bar:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        src = torch.flip(src, [0])

        optimizer.zero_grad()
        output = model(src, trg)

        output = output[1:].reshape(-1, output.shape[-1])
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

# -------------------- Start Training --------------------
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

for epoch in range(start_epoch, N_EPOCHS):
    loss = train(model, train_loader, optimizer, criterion, CLIP)
    print(f"📘 Epoch {epoch+1}/{N_EPOCHS} Loss: {loss:.4f}")

    # ✅ Save full checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt"))

    # Optional BLEU
    if (epoch + 1) % 5 == 0:
        from evaluate_bleu import evaluate_bleu
        bleu = evaluate_bleu(model, train_dataset, n_samples=100)
        print(f"🔵 Epoch {epoch+1} BLEU: {bleu:.4f}")

# ✅ Save final model weights
torch.save(model.state_dict(), "seq2seq_lstm_en_hi.pt")
print("✅ Final model saved.")
