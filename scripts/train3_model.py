import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from build_data import Vocab
import gzip
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import time
import math
import torch.optim as optim
import argparse

SOS_token = "<SOS>"
MAX_LENGTH = 40
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        weighted_query = self.Wa(query)
        weighted_keys = self.Ua(keys)
        print(f"weighted query: {weighted_query.shape}")
        print(f"weighted key: {weighted_keys.shape}")
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        print(f"Pre-scores: {scores.shape}")
        scores = scores.squeeze(2).unsqueeze(1)
        print(f"Scores: {scores.shape}")

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        print(f"Weights calculated: {weights.shape}")
        print(f"Context calculated: {context.shape}")
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, device = "cpu"):
        print(f"Encoder outputs: {encoder_outputs.shape} - device: {encoder_outputs.device}")
        print(f"Encoder_hidden: {encoder_hidden.shape} - device: {encoder_hidden.device}")
        print(f"Target tensor: {target_tensor.shape} - device: {target_tensor.device}")
        print(f"Device in forward: {device}")
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_index) # switched to SOS_index from SOS_token
        decoder_hidden = encoder_hidden.unsqueeze(dim = 1)
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        print(f"Hidden shape: {hidden.shape}")
        query = hidden
        hidden = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        print(f"Dim embedded: {embedded.shape}")
        print(f"Dim context: {context.shape}")
        input_gru = torch.cat((embedded, context), dim=2)
        print(f"INput gru: {input_gru.shape}")
        print(f"Hidden again: {hidden.shape}")
        print(f"Query shape: {query.shape}")
        output, hidden = self.gru(input_gru, hidden) # replaced self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_pairs(datafile, max_size = 10):
    data = []
    with gzip.open(datafile, "r") as file:
        i = 0
        for line in file:
            if i < max_size:
                data.append(json.loads(line))
            else:
                break
            print(f"Datapoint: {i}")
            i += 1
    return data

def get_dataloader(batch_size, train_path, tgt_vocab):
    pairs = get_pairs(train_path)
    with open(tgt_vocab, "r") as vcb:
        output_lang = Vocab("de", json.load(vcb))

    n = len(pairs)
    print(f"Data size: {n}")
    inputs = []
    targets = []

    for idx, (inp, tgt) in enumerate(pairs):
        input_tensor = torch.zeros((MAX_LENGTH, 768), dtype=torch.float)
        target_tensor = torch.zeros((MAX_LENGTH), dtype=torch.long)

        inp_tensor = torch.tensor(inp, dtype=torch.float).squeeze(dim=0)  # Ensure inp is [seq_len, 768]
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)

        seq_len = min(len(inp_tensor), MAX_LENGTH)
        target_len = min(len(tgt_tensor), MAX_LENGTH)

        input_tensor[:seq_len, :] = inp_tensor[:seq_len, :]
        target_tensor[:target_len] = tgt_tensor[:target_len]

        inputs.append(input_tensor)
        targets.append(target_tensor)

    train_data = TensorDataset(torch.stack(inputs, dim = 0), torch.stack(targets, dim = 0))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return output_lang, train_dataloader

def train_epoch(dataloader,decoder, decoder_optimizer, criterion, device):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        decoder_optimizer.zero_grad()

        encoder_hidden = input_tensor.mean(dim = 1).to(device) # across seq len
        decoder_outputs, _, _ = decoder(input_tensor.to(device), encoder_hidden.to(device), target_tensor.to(device), device)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100, device = "cpu"):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader,decoder, decoder_optimizer, criterion, device)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0



def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=768, type=int, help = "hidden size of encoder/decoder, also word vector size")
    parser.add_argument('--n_iters', default=1000, type=int, help = "total number of batches to train on")
    parser.add_argument('--tgt_vcb', help = "target vocabulary json")
    parser.add_argument('--print_every', default=10, type=int, help = "print loss info every this many training examples")
    parser.add_argument('--checkpoint_every', default=10, type=int, help = "write out checkpoint every this many batches")
    parser.add_argument('--batch_size', default=16, type=int, help = "batch_size")
    parser.add_argument('--initial_lr', default=0.001, type=float, help = "initial learning rate")
    parser.add_argument("--max_length", default = 40, nargs="?", help = "max number of tokens in generation")
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
    train_losses = []
    dev_losses = []
    bleu_scores = []
    
    # train_pairs = get_pairs(args.train_file, max_size=2000)
    # dev_pairs = get_pairs(args.dev_file, max_size = 1000)
    # silver_pairs = get_pairs(args.silver_file)
    # test_pairs = get_pairs(args.test_file)

    hidden_size = 768
    batch_size = 32

    output_lang, train_dataloader = get_dataloader(batch_size, args.train_file, args.tgt_vcb)

    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, decoder, 80, print_every=5, plot_every=5, device=device)

