import torch
import torch.nn.functional as F
from model import Encoder, Decoder, Seq2Seq
from dataset import TranslationDataset
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset to reuse vocabularies
dataset = TranslationDataset(r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt", min_freq=2)
src_vocab = dataset.src_vocab
trg_vocab = dataset.trg_vocab

# Model parameters (must match training)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
DROPOUT = 0.5

# Load trained model
encoder = Encoder(len(src_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(len(trg_vocab), EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model_path = "seq2seq_lstm_en_hi.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")
else:
    print("❌ Model file not found. Please train the model first.")
    exit()

def beam_search_translate(sentence, model, src_vocab, trg_vocab, beam_width=5, max_len=50):
    model.eval()
    tokens = sentence.lower().split()[::-1]  # reverse input like training
    indices = [src_vocab.word_to_index("<sos>")] + [src_vocab.word_to_index(tok) for tok in tokens] + [src_vocab.word_to_index("<eos>")]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)  # [src_len, 1]

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    beams = [([trg_vocab.word_to_index("<sos>")], 0.0, hidden, cell)]

    for _ in range(max_len):
        new_beams = []

        for seq, score, hidden, cell in beams:
            input_tok = torch.tensor([seq[-1]], device=DEVICE)

            with torch.no_grad():
                output, hidden_new, cell_new = model.decoder(input_tok, hidden, cell)
                log_probs = F.log_softmax(output, dim=1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                next_tok = topk_indices[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                new_seq = seq + [next_tok]
                new_beams.append((new_seq, next_score, hidden_new, cell_new))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if all(seq[-1] == trg_vocab.word_to_index("<eos>") for seq, _, _, _ in beams):
            break

    final_seq = beams[0][0]
    words = [trg_vocab.index_to_word(idx) for idx in final_seq[1:] if idx != trg_vocab.word_to_index("<eos>")]
    return ' '.join(words)

if __name__ == "__main__":
    print("\n📝 Type an English sentence to translate to Hindi.")
    print("Type 'exit' to quit.\n")

    while True:
        sentence = input("Enter English sentence (or 'exit'): ")
        if sentence.strip().lower() == "exit":
            print("👋 Exiting translation loop.")
            break

        translation = beam_search_translate(sentence, model, src_vocab, trg_vocab, beam_width=5)
        print("🔁 Hindi Translation:", translation)
        print()
