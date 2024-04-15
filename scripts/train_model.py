

import argparse
import torch
import jsonlines
import torch.nn as nn
import torch.optim as optim
import json
import os
from build_lexicon import split_lines, make_vocabs, tensors_from_pair
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import logging
import random
import time

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
    def forward(self, labels, embeddings, use_teacher_forcing=False, max_length=25): # assume inputs is batch of sequences (batch_size, sequence_len, 1)
        pred_outputs = []
        logit_outputs = []
        # Initialization
        h_n, c_n = self.get_initial_state(embeddings) # get encoder end state
        batch_size = labels.size(0)
        curr_input = self.embedding(torch.full((batch_size, 1), SOS_index)).to(labels.device) # (1, hidden_size)

        print(f"Input shape: {labels.shape}")
        print(f"curr_input shape: {labels[:, t, :].shape}")

        for t in range(max_length):
            output, (h_n, c_n) = self.lstm(curr_input, (h_n, c_n)) # expects input: (N, L, H_in)
            logits = self.output(output)
            pred = logits.argmax(dim = -1) # argmax across final output dim
            logit_outputs.append(logits)
            pred_outputs.append(pred)
            if (pred == EOS_index).all():
                break

            curr_input = labels[:, t, :] if use_teacher_forcing else pred
            curr_input = self.embedding(curr_input).view(batch_size, max_length, -1)

        batched_outputs = torch.cat(pred_outputs, dim = 1)
        batched_logits = torch.cat(logit_outputs, dim = 1)
        return batched_outputs, batched_logits

    def get_initial_state(self, embeddings):
        assert embeddings.size(1) == self.hidden_size # embeddings must have hidden_size
        embedding_tensor = torch.stack(embeddings)
        h_0 = embedding_tensor.unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        return (h_0, c_0)

# First item in pair is embedding, second is tgt_sent
def get_pairs(datafile):
    data = []
    with open(datafile, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=768, type=int, help = "hidden size of encoder/decoder, also word vector size")
    parser.add_argument('--n_iters', default=6000, type=int, help = "total number of batches to train on")
    parser.add_argument('--vcbs', help = "vocabulary json")
    parser.add_argument('--print_every', default=10, type=int, help = "print loss info every this many training examples")
    parser.add_argument('--checkpoint_every', default=10, type=int, help = "write out checkpoint every this many batches")
    parser.add_argument('--batch_size', default=128, type=int, help = "batch_size")
    parser.add_argument('--initial_learning_rate', default=0.001, type=float, help = "initial learning rate")
    parser.add_argument('--train_file', default = "train_file", help = "train data file")
    parser.add_argument('--dev_file', default = "dev_file", help= "dev data file")
    parser.add_argument('--silver_file', default = "silver_file", help= "silver data file")
    parser.add_argument('--test_file', default = "test_file", help = "test data file")
    parser.add_argument('--out_file', default = "out.txt", help = "output file for test translations")
    parser.add_argument('--load_checkpoint', default = None, nargs=1, help = "checkpoint file to start from")

    args, rest = parser.parse_known_args()

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
        vcbs = json.load(args.vcbs)
        src_vocab, tgt_vocab = vcbs["src"], vcbs["tgt"]
    

    model = LSTMDecoder(embed_dim = 768, hidden_size=768, vocab_size=tgt_vocab)
    optimizer = optim.Adam(model.params(), lr=args.initial_learning_rate)
    loss_fn = nn.NLLLoss(reduction="none")

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
            input_tensor = torch.tensor(training_pair[0])
            target_tensor = torch.tensor(training_pair[1]).view(1, -1)

            logging.info(f"Input tensor shape: {input_tensor.shape}")
            logging.info(f"Target tensor shape: {target_tensor.shape}")

            input_batch.append(input_tensor)
            target_batch.append(target_tensor)
            padded_input_batch = pad_sequence(input_batch, batch_first=True, padding_value=0)
            padded_target_batch = pad_sequence(target_batch, batch_first=True, padding_value=0)
            # input_batch_tensor = torch.cat(padded_input_batch, dim = 1)
            # target_batch_tensor = torch.cat(padded_target_batch, dim = 1)
            loss = 0
            # loss = train(padded_input_batch, padded_target_batch, encoder,
            #     decoder, optimizer, loss_fn)
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
            logging.info(f"time since start:{time.time() - start} (iter:{iter_num} iter/n_iters:{iter_num / args.n_iters * 100}) lsos_avg({print_loss_avg:.4f})")