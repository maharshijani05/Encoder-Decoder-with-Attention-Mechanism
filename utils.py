import torch
import torch.nn.functional as F

def beam_search_translate(sentence, model, src_vocab, trg_vocab, beam_width=5, max_len=50):
    model.eval()
    tokens = sentence.lower().split()[::-1]
    indices = [src_vocab.word_to_index("<sos>")] + [src_vocab.word_to_index(tok) for tok in tokens] + [src_vocab.word_to_index("<eos>")]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(model.device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    beams = [([trg_vocab.word_to_index("<sos>")], 0.0, hidden, cell)]

    for _ in range(max_len):
        new_beams = []
        for seq, score, hidden, cell in beams:
            input_tok = torch.tensor([seq[-1]], device=model.device)
            with torch.no_grad():
                output, hidden, cell = model.decoder(input_tok, hidden, cell, encoder_outputs)
                log_probs = F.log_softmax(output, dim=1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                next_tok = topk_indices[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                new_seq = seq + [next_tok]
                # ✅ Use the current hidden/cell directly (they're already updated)
                new_beams.append((new_seq, next_score, hidden, cell))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == trg_vocab.word_to_index("<eos>") for seq, _, _, _ in beams):
            break

    final_seq = beams[0][0]
    words = [trg_vocab.index_to_word(idx) for idx in final_seq[1:] if idx != trg_vocab.word_to_index("<eos>")]
    return ' '.join(words)

def translate_greedy(sentence, model, src_vocab, trg_vocab, max_len=50):
    model.eval()
    tokens = sentence.lower().split()[::-1]
    indices = [src_vocab.word_to_index("<sos>")] + [src_vocab.word_to_index(tok) for tok in tokens] + [src_vocab.word_to_index("<eos>")]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(model.device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    input_tok = torch.tensor([trg_vocab.word_to_index("<sos>")], device=model.device)
    outputs = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_tok, hidden, cell, encoder_outputs)
        pred_token = output.argmax(1).item()
        if pred_token == trg_vocab.word_to_index("<eos>"):
            break
        outputs.append(pred_token)
        input_tok = torch.tensor([pred_token], device=model.device)

    words = [trg_vocab.index_to_word(idx) for idx in outputs]
    return ' '.join(words)
