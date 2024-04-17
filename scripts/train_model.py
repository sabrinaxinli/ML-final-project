

import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import logging
import random
import time
from build_data import Vocab

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1

class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # pretrained embedding size must be the same as hidden size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, batch_first = True)
        self.output = nn.Linear(in_features = hidden_size, out_features = vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    # we only have inputs because we are implementing teacher-forcing
    def forward(self, labels, embeddings, use_teacher_forcing=False, max_length=50, device="cpu"): # assume inputs is batch of sequences (batch_size, sequence_len, 1)
        pred_outputs = []
        logit_outputs = []
        # Initialization
        print(f"label shape: {labels.shape}")
        h_n, c_n = self.get_initial_state(embeddings, device = device) # get encoder end state
        batch_size = labels.size(0)
        curr_input = self.embedding(torch.full((batch_size, 1), SOS_index, device=device)) # (1, hidden_size)

        print(f"Input shape: {labels.shape}")
        print(f"curr_input shape: {labels[:, 0].shape}")

        for t in range(max_length):
            output, (h_n, c_n) = self.lstm(curr_input, (h_n, c_n)) # expects input: (N, L, H_in)
            logits = self.output(output).squeeze(dim = 1) # convert to [batch_size, num_classes]
            pred = logits.argmax(dim = -1) # argmax across final output dim
            logit_outputs.append(logits)
            pred_outputs.append(pred)
            if (pred == EOS_index).all():
                break

            curr_input = labels[:, t] if use_teacher_forcing else pred
            curr_input = self.embedding(curr_input).view(batch_size, 1, -1) # [batch_size, seq_len = 1, hidden_size]

        batched_outputs = torch.stack(pred_outputs, dim = 1) # stack on dim = 1, to restore seq_len -> [batch_size, seq_len = 1, num_classes]
        batched_logits = torch.stack(logit_outputs, dim = 1)
        print(f"batched logits shape HERE : {batched_logits.shape}")

        # batched_output_probs = torch.nn.functional.log_softmax(batched_logits, dim = 2)
        # print(f"batched_output_probs shape: {batched_output_probs.shape}") # get to [batch_size, num_classes]
        # return batched_outputs, batched_output_probs
        return batched_outputs, batched_logits

    def get_initial_state(self, embeddings, device):
        embeddings = torch.stack(embeddings, dim = 0).to(device)
        h_0 = embeddings.unsqueeze(0) # (1, N, Hidden)
        c_0 = torch.zeros_like(h_0, device = device)
        return (h_0, c_0)

# First item in pair is embedding, second is tgt_sent
def get_pairs(datafile):
    data = []
    with open(datafile, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data

def zero_out_post_eos(probs, outputs):
    mask = (outputs == EOS_index).cumsum(dim = 1) <= 1 # [batch_size, seq_len]
    mask = mask.unsqueeze(dim = 2) # [batch_size, seq_len, 1]
    return probs * mask.float()

def pad_sequence_tensor_to_max(sequences, max_length, pad_value = 0):
    current_length = sequences.size(1)
    padding_size = max(0, max_length - current_length)
    padded_sequences = F.pad(sequences, (0, 0, 0, padding_size), "constant", value=pad_value)
    return padded_sequences

def pad_sequence_list_to_max(sequences, max_length, pad_value = 0):
    padded_sequences = [F.pad(seq, (0, max(0, max_length - len(seq))), "constant", value=pad_value) for seq in sequences]
    padded_tensor = torch.stack(padded_sequences, dim = 0) # restore batch_size
    return padded_tensor


# Parameters: input_batch -> Lst[tensors], target_batch -> tensor
def train(model, optimizer, loss_fn, input_batch, target_batch, max_length, device = "cpu"):
    
    model.train()
    model.to(device)

    target_batch = target_batch.to(device)

    batched_outputs, batched_output_probs = model(labels = target_batch, embeddings = input_batch, device = device)
    cleaned_probs = zero_out_post_eos(batched_output_probs, batched_outputs)
    if cleaned_probs.size(1) < max_length:
        cleaned_probs = pad_sequence_tensor_to_max(cleaned_probs, max_length, pad_value = SOS_index)

    total_loss = 0
    for t in range(max_length):
        total_loss += loss_fn(cleaned_probs[:, t, :], target_batch[:, t])
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# logging.info(f"Batched outputs shape: {batched_output_probs.shape}")
    # logging.info(f"Batched output probs shape: {batched_output_probs.shape}")
    # print(f"Target batch shape: {target_batch.shape}")
    # logging.info(f"Cleaned probs shape : {cleaned_probs.shape}")
    # logging.info(f"Target batch : {target_batch.shape}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=768, type=int, help = "hidden size of encoder/decoder, also word vector size")
    parser.add_argument('--n_iters', default=1000, type=int, help = "total number of batches to train on")
    parser.add_argument('--vcbs', help = "vocabulary json")
    parser.add_argument('--print_every', default=10, type=int, help = "print loss info every this many training examples")
    parser.add_argument('--checkpoint_every', default=10, type=int, help = "write out checkpoint every this many batches")
    parser.add_argument('--batch_size', default=16, type=int, help = "batch_size")
    parser.add_argument('--initial_lr', default=0.001, type=float, help = "initial learning rate")
    parser.add_argument("--max_length", default = 50, nargs="?", help = "max number of tokens in generation")
    parser.add_argument('--train_file', default = "train_file", help = "train data file")
    parser.add_argument('--dev_file', default = "dev_file", help= "dev data file")
    parser.add_argument('--silver_file', default = "silver_file", help= "silver data file")
    parser.add_argument('--test_file', default = "test_file", help = "test data file")
    parser.add_argument('--out_file', default = "out.txt", help = "output file for test translations")
    parser.add_argument('--load_checkpoint', default = None, nargs="?", help = "checkpoint file to start from")

    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = "cpu"

    print(f"Device: {device}")
    iter_num = 0
    
    train_pairs = get_pairs(args.train_file)
    dev_pairs = get_pairs(args.dev_file)
    silver_pairs = get_pairs(args.silver_file)
    test_pairs = get_pairs(args.test_file)

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint)
        iter_num = state["iter_num"]
        src_vocab = state["src_vocab"]
        tgt_vocab = state["tgt_vocab"]
        
    else:
        iter_num = 0
        with open(args.vcbs, "r") as vocab_files:
            vcbs = json.load(vocab_files)
        src_vocab, tgt_vocab = Vocab(vocab_dict = vcbs["src"]), Vocab(vocab_dict = vcbs["tgt"])
    

    model = LSTMDecoder(embed_dim = 768, hidden_size=768, vocab_size=tgt_vocab.n_words).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    # loss_fn = nn.NLLLoss(reduction="none")
    loss_fn = nn.CrossEntropyLoss(ignore_index = SOS_index)

    if args.load_checkpoint:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["opt_state"])

    start = time.time()
    print_loss_total = 0
    while iter_num < args.n_iters:
        iter_num += 1
        input_batch = []
        target_batch = []
        for i in range(args.batch_size):
            training_pair = random.choice(train_pairs)
            input_tensor = torch.tensor(training_pair[0], dtype = torch.float).squeeze(dim = 0).to(device) # needs to be of shape [seq_len]
            target_tensor = torch.tensor(training_pair[1], dtype = torch.long).to(device)
            if target_tensor.size(0) > args.max_length:
                print("Truncated")
            target_tensor = target_tensor[:args.max_length] # truncate
            input_batch.append(input_tensor)
            target_batch.append(target_tensor)
            
        # No need to pad input_batch because these are fixed-length embeds
        # Pad target sentences to max_length
        padded_target_batch = pad_sequence_list_to_max(target_batch, max_length = args.max_length, pad_value = SOS_index)
        loss = train(model, optimizer, loss_fn, input_batch, padded_target_batch, max_length = args.max_length, device = device)
        
        print_loss_total += loss
        if iter_num % args.checkpoint_every == 0:
            state = {"iter_num": iter_num,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                }
            filename = f'state_{iter_num:010d}.pt'
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info(f"time since start:{time.time() - start} (iter:{iter_num} iter/n_iters:{iter_num / args.n_iters * 100}) loss_avg({print_loss_avg:.4f})")