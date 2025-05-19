import torch
from model import Encoder, Decoder, Seq2Seq
from dataset import TranslationDataset, collate_fn
from torch.nn.functional import softmax
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset to reuse vocab
dataset = TranslationDataset(r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt", min_freq=2)
src_vocab = dataset.src_vocab
trg_vocab = dataset.trg_vocab

# Hyperparameters (must match training)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.5

# Load model
encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(trg_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model.load_state_dict(torch.load("seq2seq_lstm_en_hi.pt", map_location=DEVICE))
model.eval()

def translate_sentence(sentence, model, src_vocab, trg_vocab, max_len=50):
    model.eval()
    tokens = sentence.lower().split()[::-1]  # reverse input
    indices = [src_vocab.word_to_index("<sos>")] + [src_vocab.word_to_index(tok) for tok in tokens] + [src_vocab.word_to_index("<eos>")]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)  # [src_len, 1]

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    input_tok = torch.tensor([trg_vocab.word_to_index("<sos>")], device=DEVICE)
    outputs = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_tok, hidden, cell)
            pred_token = output.argmax(1).item()
        if pred_token == trg_vocab.word_to_index("<eos>"):
            break
        outputs.append(trg_vocab.index_to_word(pred_token))
        input_tok = torch.tensor([pred_token], device=DEVICE)

    return ' '.join(outputs)

if __name__ == "__main__":
    while True:
        sentence = input("Enter English sentence (or 'exit'): ")
        if sentence.lower() == 'exit':
            break
        hindi = translate_sentence(sentence, model, src_vocab, trg_vocab)
        print("🔁 Hindi Translation:", hindi)

