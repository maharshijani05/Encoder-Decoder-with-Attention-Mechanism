import spacy
from indicnlp.tokenize import indic_tokenize
from tqdm import tqdm

# Load English tokenizer
nlp_en = spacy.load('en_core_web_sm')

def tokenize_en(sentence):
    return [tok.text for tok in nlp_en.tokenizer(sentence)]

def tokenize_hi(sentence):
    # Indic NLP's trivial tokenizer for Hindi
    return indic_tokenize.trivial_tokenize(sentence, lang='hi')

def preprocess_parallel_corpus(en_file, hi_file, output_file, max_sentences=None):
    with open(en_file, 'r', encoding='utf-8') as f_en, open(hi_file, 'r', encoding='utf-8') as f_hi, open(output_file, 'w', encoding='utf-8') as f_out:
        en_lines = f_en.readlines()
        hi_lines = f_hi.readlines()
        
        for i, (en_sent, hi_sent) in enumerate(tqdm(zip(en_lines, hi_lines))):
            if max_sentences and i >= max_sentences:
                break
            
            en_tokens = tokenize_en(en_sent.strip())
            hi_tokens = tokenize_hi(hi_sent.strip())
            
            if not en_tokens or not hi_tokens:
                continue

            # Join tokens back as space separated strings
            en_tokenized = ' '.join(en_tokens)
            hi_tokenized = ' '.join(hi_tokens)
            
            # Save tab-separated tokenized sentence pair
            f_out.write(f"{en_tokenized}\t{hi_tokenized}\n")

if __name__ == "__main__":
    en_path = 'E:\Encoder_Decoder_Scratch\parallel-n/IITB.en-hi.en'
    hi_path = 'E:\Encoder_Decoder_Scratch\parallel-n/IITB.en-hi.hi'
    output_path = 'E:\Encoder_Decoder_Scratch/tokenized_en_hi.txt'
    
    preprocess_parallel_corpus(en_path, hi_path, output_path, max_sentences=500000)

