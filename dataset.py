import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

class Vocab:
    def __init__(self, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.freqs = Counter()
        self.build_called = False

    def build_vocab(self, sentences):
        for sentence in sentences:
            self.freqs.update(sentence)

        idx = 0
        for token in SPECIAL_TOKENS:
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1

        for word, freq in self.freqs.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.build_called = True

    def __len__(self):
        return len(self.word2idx)

    def word_to_index(self, word):
        return self.word2idx.get(word, self.word2idx[UNK_TOKEN])

    def index_to_word(self, idx):
        return self.idx2word.get(idx, UNK_TOKEN)

    def sentence_to_indices(self, sentence):
        return [self.word_to_index(SOS_TOKEN)] + [self.word_to_index(w) for w in sentence] + [self.word_to_index(EOS_TOKEN)]

    def indices_to_sentence(self, indices):
        return [self.index_to_word(idx) for idx in indices]

class TranslationDataset(Dataset):
    def __init__(self, tokenized_file, src_vocab=None, trg_vocab=None, min_freq=2, max_len=50):
        self.src_sentences = []
        self.trg_sentences = []
        self.max_len = max_len

        with open(tokenized_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # skip malformed line

            src, trg = parts
            src_tokens = src.strip().split()
            trg_tokens = trg.strip().split()

            if 0 < len(src_tokens) <= max_len and 0 < len(trg_tokens) <= max_len:
                self.src_sentences.append(src_tokens)
                self.trg_sentences.append(trg_tokens)

        # Build vocabularies if not provided
        if src_vocab is None:
            self.src_vocab = Vocab(min_freq)
            self.src_vocab.build_vocab(self.src_sentences)
        else:
            self.src_vocab = src_vocab

        if trg_vocab is None:
            self.trg_vocab = Vocab(min_freq)
            self.trg_vocab.build_vocab(self.trg_sentences)
        else:
            self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_indices = self.src_vocab.sentence_to_indices(self.src_sentences[idx])
        trg_indices = self.trg_vocab.sentence_to_indices(self.trg_sentences[idx])
        return torch.tensor(src_indices), torch.tensor(trg_indices)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    trg_lens = [len(x) for x in trg_batch]

    src_pad_len = max(src_lens)
    trg_pad_len = max(trg_lens)

    padded_src = torch.zeros(len(batch), src_pad_len).long()
    padded_trg = torch.zeros(len(batch), trg_pad_len).long()

    for i, (src_seq, trg_seq) in enumerate(batch):
        padded_src[i, :len(src_seq)] = src_seq
        padded_trg[i, :len(trg_seq)] = trg_seq

    return padded_src.transpose(0,1), padded_trg.transpose(0,1)  # [seq_len, batch_size]

if __name__ == "__main__":
    dataset = TranslationDataset(r"E:\Encoder_Decoder_Scratch\tokenized_en_hi.txt", min_freq=2)
    print(f"✅ Dataset size: {len(dataset)}")
    print(f"🔤 Src vocab size: {len(dataset.src_vocab)}")
    print(f"🔠 Trg vocab size: {len(dataset.trg_vocab)}")

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    for src_batch, trg_batch in loader:
        print(f"📦 Source batch shape: {src_batch.shape}")
        print(f"📦 Target batch shape: {trg_batch.shape}")
        break
