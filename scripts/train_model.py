

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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=256, type=int,
            help='hidden size of encoder/decoder, also word vector size')
    parser.add_argument('--n_iters', default=6000, type=int,
            help='total number of examples to train on')
    parser.add_argument('--print_every', default=10, type=int,
            help='print loss info every this many training examples')
    parser.add_argument('--checkpoint_every', default=5, type=int,
            help='write out checkpoint every this many training examples')
    parser.add_argument('--initial_learning_rate', default=0.005, type=float,
            help='initial learning rate')
    parser.add_argument('--train_file', default='data/fren.train.bpe',
            help='training file. each line should have a source sentence,' +
                'followed by "|||", followed by a target sentence')
    parser.add_argument('--dev_file', default='data/fren.dev.bpe',
            help='dev file. each line should have a source sentence,' +
                'followed by "|||", followed by a target sentence')
    parser.add_argument('--test_file', default='data/fren.test.bpe',
            help='test file. each line should have a source sentence,' +
                'followed by "|||", followed by a target sentence' +
                ' (for test, target is ignored)')
    parser.add_argument('--out_file', default='out.txt',
            help='output file for test translations')
    parser.add_argument('--load_checkpoint', default = "test", nargs=1,
            help='checkpoint file to start from')

    args, rest = parser.parse_known_args()

    iter_num = 0
    src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                       args.tgt_lang,
                       args.train_file)
    

    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(reduction="none")

    while iter_num < args.n_iters:
        iter_num += 1
        input_batch = []
        target_batch = []
        for i in range(args.batch_size):
            training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            input_batch.append(input_tensor)
            target_batch.append(target_tensor)
            padded_input_batch = pad_sequence(input_batch, batch_first=False, padding_value=0)
            padded_target_batch = pad_sequence(target_batch, batch_first=False, padding_value=0)
            # input_batch_tensor = torch.cat(padded_input_batch, dim = 1)
            # target_batch_tensor = torch.cat(padded_target_batch, dim = 1)
            loss = train(padded_input_batch, padded_target_batch, encoder,
                decoder, optimizer, criterion)
            # print(f"Loss on iteration {iter_num} : {loss}")
            # print(translate(encoder, decoder, "c est pour vous que je suis", src_vocab, tgt_vocab))
            print_loss_total += loss
        if iter_num % args.checkpoint_every == 0:
        state = {'iter_num': iter_num,
            'enc_state': encoder.state_dict(),
            'dec_state': decoder.state_dict(),
            'opt_state': optimizer.state_dict(),
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            }
        filename = 'state_%010d.pt' % iter_num
        torch.save(state, filename)
        logging.debug('wrote checkpoint to %s', filename)