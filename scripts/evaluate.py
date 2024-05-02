from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import gzip
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
OOV_token = "<OOV>"
SOS_index = 0
EOS_index = 1
OOV_index = 2
MAX_LENGTH = 41

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code=None, vocab_dict=None):
        if vocab_dict:
            self.lang_code = vocab_dict["lang_code"]
            self.word2index = vocab_dict["word2index"]
            self.word2count = vocab_dict["word2count"]
            self.index2word = vocab_dict["index2word"]
            self.n_words = vocab_dict["n_words"]
        else:
            self.lang_code = lang_code
            self.word2index = {OOV_token : OOV_index}
            self.word2count = {}
            self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, OOV_index: OOV_token}
            self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def to_dict(self):
        return {
            "lang_code": self.lang_code,
            "word2index": self.word2index,
            "word2count": self.word2count,
            "index2word": self.index2word,
            "n_words": self.n_words
        }

# From PyTorch
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# From PyTorch: Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

# Get pairs from json file
def get_pairs(datafile, max_size = 100000):
    data = []
    with open(datafile, "r") as file:
        i = 0
        for line in file:
            if i < max_size:
                data.append(json.loads(line))
            else:
                break
            print(f"Datapoint: {i}")
            i += 1
    return data

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, OOV_index) for word in sentence.split(" ")]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(src_vocab, pair[0])
    target_tensor = tensorFromSentence(tgt_vocab, pair[1])
    return (input_tensor, target_tensor)

def create_vocabs(filepath, lang1, lang2, max_size = 5000, reverse=False):
    print("Begin reading lines")
    pairs = get_pairs(filepath, max_size = max_size)

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Vocab(lang2)
        output_lang = Vocab(lang1)
    else:
        input_lang = Vocab(lang1)
        output_lang = Vocab(lang2)

    return input_lang, output_lang, pairs

# def filterPair(p):
#     print(type(p))
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH


# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]

def prepareData(filepath, lang1, lang2, max_size, reverse=False):
    input_lang, output_lang, pairs = create_vocabs(filepath, lang1, lang2, max_size = max_size, reverse = reverse)
    print(f"Found {len(pairs)} sentence pairs")
    # pairs = filterPairs(pairs)
    # print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Begin adding sentences")
    for pair in pairs:
        input_lang.add_sentence("".join(pair[0]))
        output_lang.add_sentence("".join(pair[1]))
    print("Number of words:")
    print(input_lang.lang_code, input_lang.n_words)
    print(output_lang.lang_code, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, (hidden_state, cell_state) = self.lstm(embedded)
        return output, (hidden_state, cell_state)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.query_weight = nn.Linear(hidden_size, hidden_size)
        self.key_weights = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.V(torch.tanh(self.query_weight(query) + self.key_weights(keys)))
        # print(f"scores shape: {scores.shape}")
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, device="cpu"):
        # Initialize parameters for model
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_index).to(device)
        # print(type(encoder_hidden))
        # print(len(encoder_hidden))
        hidden_state, cell_state = (encoder_hidden, torch.zeros_like(encoder_hidden, device = device))

        # Initialize for sequence
        decoder_outputs = []
        attentions = []

        # Decode max_length times
        for i in range(MAX_LENGTH):
            decoder_output, hidden_state, cell_state, attn_weights = self.forward_step(
                decoder_input, hidden_state, cell_state, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            # Teacher forcing
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(dim = 1) # unsqueeze to get [batch_size, seq_len = 1]
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        # Combine all outputs into one tensor and convert to log probs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # Get attentions as one tensor
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, hidden_state, cell_state, attentions
    
    # Forward step for one token
    def forward_step(self, input, hidden_state, cell_state, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden_state.permute(1, 0, 2)
        context, attention_weights = self.attention(query, encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
        output = self.out(output)

        return output, hidden_state, cell_state, attention_weights
    


def get_dataloader(filepath, batch_size, max_size, reverse):
    input_vocab, output_vocab, pairs = prepareData(filepath, 'eng', 'de', max_size = max_size, reverse = reverse)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (input, target) in enumerate(pairs):
        src_ids = indexesFromSentence(input_vocab, input)
        tgt_ids = indexesFromSentence(output_vocab, target)
        src_ids.append(EOS_index)
        tgt_ids.append(EOS_index)
        input_ids[idx, :len(src_ids)] = src_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))
    print(f"Train data: {len(train_data)}")
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_vocab, output_vocab, train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, (encoder_hidden_state, encoder_cell_state) = encoder(input_tensor)
        decoder_outputs, _, _, attentions = decoder(encoder_outputs, encoder_hidden_state, target_tensor, device)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f"{asMinutes(s)} (- {asMinutes(rs)})"

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100:.0f}%) {print_loss_avg:.4f}')

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points, savepath):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.savefig(savepath)
    plt.plot(points)

import matplotlib.pyplot as plt

def plot_data(points, interval, xlabel, ylabel, title, savepath):
    print(f"SIZE: {len(points) * interval}")
    epochs = list(range(0, len(points) * interval, interval))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, points)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(savepath)
    plt.show()



def evaluate(encoder, decoder, sentence, src_vcb, tgt_vcb, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(src_vcb, sentence)

        encoder_outputs, (encoder_hidden_state, encoder_cell_state) = encoder(input_tensor)
        decoder_outputs, _, _, decoder_attention = decoder(encoder_outputs, encoder_hidden_state, None, device = device)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_index:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(tgt_vcb.index2word[idx.item()])
    return decoded_words, decoder_attention

def evaluate_on_set(encoder, decoder, pairs, src_vcb, tgt_vcb):
    bleu_scores = []
    for (src, tgt) in pairs:
        decoded_words, decoder_attention = evaluate(encoder, decoder, src, src_vcb, tgt_vcb, device)
        print(f"Label: {tgt}")
        print(f"Prediction: {decoded_words}")
        bleu_score = sentence_bleu([tgt.split()], decoded_words)
        bleu_scores.append(bleu_score)
    
    return sum(bleu_scores) / len(bleu_scores)


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default = "eng-de-train.jsonlines", help="File that contains src_tgt sentence together in jsonlines format")
    parser.add_argument("--dev_file", default = "eng-de-dev.jsonlines", help = "Dev file, also jsonlines")
    parser.add_argument("--test_file", default = "eng-de-test.jsonlines",  help = "Test file, also jsonlines")
    parser.add_argument("--src_lang", default="en", help='Source (input) language code, e.g. "en"')
    parser.add_argument("--tgt_lang", default="de", help="Target (output) language code, e.g. 'de'")
    # parser.add_argument("--src_vcb", help="Source vocab json output file")
    # parser.add_argument("--tgt_vcb", help="target vocab json output file")

    parser.add_argument("--loss_graphs", default = ["train_loss.png"], nargs = 1, help = "")
    parser.add_argument("--score_graphs", default = ["dev_scores.png", "test_scores.png"], nargs = 2, help = "")

    parser.add_argument("--max_samples", nargs = "?", type = int, help="Maximum number of datapoints")
    parser.add_argument("--batch_size", default = 32, type = int, help='output file for test translations')
    parser.add_argument("--hidden_size", default = 256, type = int, help='output file for test translations')
    parser.add_argument("--epochs", default = 101, type = int, help = "Num epochs")
    parser.add_argument("--lr", default = 0.01, type = float, help = "Learning rate")
    parser.add_argument("--print_every", default = 1, type = int, help = "Print every epochs")
    parser.add_argument("--plot_every", default = 1, type = int, help = "Every n iterations, add a point for plotting")
    parser.add_argument("--check_every", default = 1, type = int, help = "Check every epochs")
    parser.add_argument("--test_every", default = 1, type = int, help = "How often to run dev and test sets")
    parser.add_argument("--load_checkpoint", default = "./encoder_decoder_lstm_100.pt")

    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    src_vocab, tgt_vocab, train_dataloader = get_dataloader(args.train_file, args.batch_size, max_size = 40000, reverse = False)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words).to(device)

    dev_set = get_pairs(args.dev_file, max_size = 2000)
    test_set = get_pairs(args.test_file, max_size = 2000)
    train_set = get_pairs(args.train_file, max_size = 2000)

    

    # train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint)
        encoder.load_state_dict(state["encoder_model_state"])
        decoder.load_state_dict(state["decoder_model_state"])
        

    train_score = evaluate_on_set(encoder, decoder, train_set, src_vocab, tgt_vocab)
    print(f"Train score: {train_score}")
