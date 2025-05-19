import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataset import TranslationDataset
from model import Encoder, Decoder, Seq2Seq
from utils import beam_search_translate
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load vocab + model
dataset = TranslationDataset(r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt", min_freq=2, max_len=50)
src_vocab = dataset.src_vocab
trg_vocab = dataset.trg_vocab

EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.5

encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(trg_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
# model.load_state_dict(torch.load("seq2seq_lstm_en_hi.pt", map_location=DEVICE))
model.eval()

from nltk.tokenize import word_tokenize

def evaluate_bleu(model, dataset, n_samples=100):
    smoothie = SmoothingFunction().method4
    total_bleu = 0
    count = 0

    for _ in range(n_samples):
        idx = random.randint(0, len(dataset) - 1)
        src_tensor, trg_tensor = dataset[idx]

        src_tokens = dataset.src_sentences[idx]
        trg_tokens = dataset.trg_sentences[idx]

        src_sentence = ' '.join(src_tokens)
        trg_sentence = ' '.join(trg_tokens)

        prediction = beam_search_translate(src_sentence, model, src_vocab, trg_vocab)
        candidate = word_tokenize(prediction)
        reference = [trg_tokens]

        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        total_bleu += bleu
        count += 1

    avg_bleu = total_bleu / count
    return avg_bleu

if __name__ == "__main__":
    score = evaluate_bleu(model, dataset, n_samples=100)
    print(f"\n🔵 Average BLEU Score (100 samples): {score:.4f}")
