import argparse
from train_model import LSTMDecoder
from build_data import get_batches
import json
import torch
import logging
# import torch.optim as optim

MAX_TOKS = 512

def embed_batched_input(model, tokenizer, src_batch, device = "cpu"):
    # Pass source sentences through BERT
    tokens = tokenizer(src_batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TOKS)
    bert_output = model(input_ids = tokens["input_ids"].to(device),
                        attention_mask = tokens["attention_mask"].to(device),
                        token_type_ids = tokens["token_type_ids"].to(device),
                        output_hidden_states = True)
    
    # Grab CLS token as sent representation
    bert_hidden_states = bert_output["hidden_states"]
    cls_token_batch = bert_hidden_states[-1][:,0,:]
    
    # Split batch into list of source sentence embeddings
    s_embeddings = torch.split(cls_token_batch, split_size_or_sections=1, dim=0)
    return s_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", help="source bitext should have one sentence per line")
    parser.add_argument("--tgt_path", help="tgt bitext should have one sentence per line, same length as src_bitext")
    parser.add_argument("--model_path", help = "Trained model path")
    parser.add_argument("--batch_size", type = int, help = "Trained model path")
    args, rest = parser.parse_known_args()

    state = torch.load(args.model_path)
    src_vocab = state["src_vocab"]
    tgt_vocab = state["tgt_vocab"]
    model = LSTMDecoder(embed_dim = 768, hidden_size=768, vocab_size=tgt_vocab)
    model.load_state_dict(state["model_state"])

    logging.info("Begin translation")

    with open(args.src_path, "r", encoding="utf-8") as src_text, open(args.tgt_path, "w", encoding="utf-8") as tgt_text:
        batches = get_batches(list(src_text), args.batch_size)
        for batch in batches:
            embeddings = embed_batched_input(batch)
            outputs = model(embeddings)
            for output in outputs:
                output_sentence = [tgt_vocab.index2word[idx] for idx in output]
                tgt_text.write(output_sentence + "\n")

    
    
    