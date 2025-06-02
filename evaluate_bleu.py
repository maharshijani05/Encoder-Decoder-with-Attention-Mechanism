import torch
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from dataset import TranslationDataset
from model import Encoder, Decoder, Seq2Seq, Attention
# from utils import beam_search_translate
from utils import translate_greedy as translate


# -------------------- Config --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_11.pt"
TOKENIZED_FILE = r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt"

# -------------------- Load Dataset --------------------
dataset = TranslationDataset(TOKENIZED_FILE, min_freq=2, max_len=50)
src_vocab = dataset.src_vocab
trg_vocab = dataset.trg_vocab

# -------------------- Model Setup --------------------
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.5

# ✅ Define attention before using it in decoder
attention = Attention(HID_DIM)

encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(trg_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attention)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# ✅ Load trained weights
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -------------------- BLEU Evaluation --------------------
def evaluate_bleu(model, dataset, n_samples=1000,print_samples=5):
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    printed = 0

    for _ in range(n_samples):
        idx = random.randint(0, len(dataset) - 1)
        src_tensor, trg_tensor = dataset[idx]

        src_tokens = dataset.src_sentences[idx]
        trg_tokens = dataset.trg_sentences[idx]

        src_sentence = ' '.join(src_tokens)
        trg_sentence = ' '.join(trg_tokens)
        # prediction = beam_search_translate(src_sentence, model, src_vocab, trg_vocab)
        prediction = translate(src_sentence, model, src_vocab, trg_vocab)
        candidate = word_tokenize(prediction)
        reference = [trg_tokens]

        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        total_bleu += bleu

        if printed < print_samples:
            print("🔹 Source:   ", src_sentence)
            print("🔹 Target:   ", trg_sentence)
            print("🔹 Predicted:", prediction)
            print("------------")
            printed += 1

    avg_bleu = total_bleu / n_samples
    return avg_bleu

# -------------------- Run --------------------
if __name__ == "__main__":
    score = evaluate_bleu(model, dataset, n_samples=1000,print_samples=5)
    print(f"\n🔵 Average BLEU Score (1000 samples): {score:.4f}")
