

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
import nltk
from nltk.translate.bleu_score import sentence_bleu
import gzip

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
    def forward(self, embeddings, labels, use_teacher_forcing=False, max_length=50, device="cpu"): # assume inputs is batch of sequences (batch_size, sequence_len, 1)
        pred_outputs = []
        logit_outputs = []

        # Initialization
        h_n, c_n = self.get_initial_state(embeddings, device = device) # get encoder end state
        batch_size = labels.size(0)
        curr_input = self.embedding(torch.full((batch_size, 1), SOS_index, device=device)) # (1, hidden_size)

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
        return batched_outputs, batched_logits

    def get_initial_state(self, embeddings, device):
        embeddings = torch.stack(embeddings, dim = 0).to(device)
        h_0 = embeddings.unsqueeze(0) # (1, N, Hidden)
        c_0 = torch.zeros_like(h_0, device = device)
        return (h_0, c_0)
    
class AttentionLSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, dropout=0.1):
        super().__init__()

        # pretrained embedding size is the same as embed_size (note: different from above!)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        contextualized_input_size = embed_dim * 2
        self.lstm = nn.LSTM(input_size = contextualized_input_size, hidden_size = hidden_size, batch_first = True)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
        self.output = nn.Linear(in_features = embed_dim * 3, out_features = vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder_attention = nn.Linear(in_features = embed_dim, out_features = hidden_size)
        self.decoder_attention = nn.Linear(in_features = hidden_size, out_features = hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    # we only have inputs because we are implementing teacher-forcing
    def forward(self, embeddings, labels, use_teacher_forcing=False, max_length=50, device="cpu"): # assume inputs is batch of sequences (batch_size, sequence_len, 1)
        pred_outputs = []
        logit_outputs = []
        batch_size = len(embeddings) # get first dim as batch_size

        # Initialization
        h_n, c_n = self.get_initial_state(batch_size, device = device) # init
        curr_input = self.dropout(self.embedding(torch.full((batch_size, 1), SOS_index, device=device))) # [batch_size, 1, embed_dim]

        for t in range(max_length):
            # Calculate attention and create attention-based input
            encoder_embeddings_tensor, attention_scores = self.calculate_attention_scores(embeddings, h_n, device) # [batch_size, seq_len]
            attention_weights = F.softmax(attention_scores, dim = 1).unsqueeze(dim = 2) # [batch_size, seq_len, 1]
            context_vector = torch.sum(attention_weights * encoder_embeddings_tensor, dim = 1).unsqueeze(dim = 1) # weighted embeds -> [batch_size, 1, embed_dim]
            # print(f"Context vector shape: {context_vector.shape}")
            # print(f"Curr input shape: {curr_input.shape}")
            weighted_embeddings_and_input = torch.cat((context_vector, curr_input), dim = 2)

            # Pass through lstm
            lstm_output, (h_n, c_n) = self.lstm(weighted_embeddings_and_input, (h_n, c_n)) # expects input: (N, L=1, H_in), output: [N, L=1, hidden_size]

            # Combine lstm_output with input embedding and weighted encoder embeddings before feeding through output
            # print(f"lstm_output : {lstm_output.shape}")
            # print(f"weighted_embeddings_and_input: {weighted_embeddings_and_input.shape}")
            output_vector = torch.cat((lstm_output, weighted_embeddings_and_input), dim = 2)
            logits = self.output(output_vector).squeeze(dim = 1) # convert to [batch_size, num_classes]
            pred = logits.argmax(dim = -1) # argmax across final output dim
            logit_outputs.append(logits)
            pred_outputs.append(pred)
            if (pred == EOS_index).all():
                break

            # Feed output back through embedding
            curr_input = labels[:, t] if use_teacher_forcing else pred
            curr_input = self.embedding(curr_input).view(batch_size, 1, -1) # [batch_size, seq_len = 1, embed_dim]

        batched_outputs = torch.stack(pred_outputs, dim = 1) # stack on dim = 1, to restore seq_len -> [batch_size, seq_len = 1, num_classes]
        batched_logits = torch.stack(logit_outputs, dim = 1)
        return batched_outputs, batched_logits
    
    def calculate_attention_scores(self, encoder_embeddings, decoder_hidden, device = "cpu"):
        # Input sizes:
            # encoder_embeddings: List: List: embeddings
            # decoder_hidden: [batch_size, decoder_dim]
        # Output size:
            # scores: [batch_size, seq_len, hidden_size]
        max_seq_len = max([seq.size(0) for seq in encoder_embeddings])
        encoder_embeddings_tensor, mask = pad_sequence_with_mask(encoder_embeddings, max_seq_len, device) # [batch_size, padded_seq_len, embed_size]
        enc_proj = self.encoder_attention(encoder_embeddings_tensor) # [batch_size, padded_seq_len, hidden_size]
        dec_proj = self.decoder_attention(decoder_hidden).squeeze(dim = 0) # [batch_size, hidden_size]

        dec_proj = dec_proj.unsqueeze(dim = 1).expand_as(enc_proj) # expand to [batch_size, seq_len, hidden_size]

        tanh_result = torch.tanh(enc_proj + dec_proj)
        pre_scores = self.v * tanh_result
        scores = torch.sum(pre_scores, dim = 2)
        scores.masked_fill_(~mask, float("-inf"))
        return encoder_embeddings_tensor, scores
    
    # def get_initial_state(self, batch_size, device):
    #     h_0 = torch.zeros((1, batch_size, self.hidden_size), device = device)
    #     c_0 = torch.zeros((1, batch_size, self.hidden_size), device = device)
    #     return (h_0, c_0)
    
    def get_initial_state(self, batch_size, device):
        h_0 = torch.nn.init.orthogonal_(torch.empty(1, batch_size, self.hidden_size, device=device))
        c_0 = torch.nn.init.orthogonal_(torch.empty(1, batch_size, self.hidden_size, device=device))
        return (h_0, c_0)


# First item in pair is embedding, second is tgt_sent
def get_pairs(datafile, max_size = 100000):
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

def zero_out_post_eos(probs, outputs):
    mask = (outputs == EOS_index).cumsum(dim = 1) <= 1 # [batch_size, seq_len]
    seq_lengths = mask.sum(dim = 1)
    mask = mask.unsqueeze(dim = 2) # [batch_size, seq_len, 1]
    return probs * mask.float(), seq_lengths

# def switch_out_post_eos(logits, outputs):
#     mask = (outputs == EOS_index).cumsum(dim = 1) <= 1 # [batch_size, seq_len]
#     seq_lengths = mask.sum(dim = 1) # up to and including EOS token
#     masked_outputs = torch.where(mask, outputs, torch.full_like(outputs, EOS_index)) # [batch_size, seq_len, 1]
#     masked_logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
#     masked_logits = masked_logits[:, :, EOS_index] = 1
#     return masked_outputs, seq_lengths

def switch_out_post_eos(logits, outputs):
    # Create a mask that is True up to and including the first occurrence of the EOS token
    mask = (outputs == EOS_index).cumsum(dim=1) <= 1
    seq_lengths = mask.sum(dim=1)


    masked_outputs = torch.where(mask, outputs, torch.full_like(outputs, EOS_index))
    masked_logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

    # Create mask specifically for the EOS_index where the first mask is false
    eos_mask = ~mask
    print(f"EOS mask: {eos_mask.shape
    eos_values = torch.full((logits.shape[0], logits.shape[1]), float('inf'), device=logits.device)
    masked_logits[:, :, EOS_index] = torch.where(eos_mask, eos_values, masked_logits[:, :, EOS_index])

    return masked_outputs, seq_lengths, masked_logits

def pad_sequence_tensor_to_max(sequences, max_length, pad_value = 0):
    current_length = sequences.size(1)
    padding_size = max(0, max_length - current_length)
    padded_sequences = F.pad(sequences, (0, 0, 0, padding_size), "constant", value=pad_value)
    return padded_sequences

def pad_sequence_list_to_max(sequences, max_length, pad_value = 0):
    padded_sequences = [F.pad(seq, (0, max(0, max_length - len(seq))), "constant", value=pad_value) for seq in sequences]
    padded_tensor = torch.stack(padded_sequences, dim = 0) # restore batch_size
    return padded_tensor

def pad_sequence_with_mask(sequences, max_length, device = "cpu"):
    # Sequences is a list of tensors [seq_len, 768]
    padded_sequences = []
    masks = []
    for seq in sequences:
        pad_length = max_length - len(seq)
        pad_tensor = torch.zeros(pad_length, seq.size(1), device = device)
        
        # Concat along seq_len dim
        padded_seq = torch.cat([seq, pad_tensor], dim=0)
        padded_sequences.append(padded_seq)
        
        # Create mask (1 for real data, 0 for padding)
        mask = torch.cat([torch.ones(len(seq), dtype=torch.bool, device = device), torch.zeros(pad_length, dtype=torch.bool, device = device)])
        masks.append(mask)

    # Stack all sequences and masks
    padded_sequences = torch.stack(padded_sequences).to(device)
    masks = torch.stack(masks).to(device)

    return padded_sequences, masks


def ids_to_sentence(output, vocab):
    if type(output) != list:
        output = output.tolist()
    return [vocab.index2word.get(str(idx), "<unk>") for idx in output if idx not in (0, 1)]

def check_for_nans(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")

# Parameters: input_batch -> Lst[tensors], target_batch -> tensor
def train(model, optimizer, loss_fn, input_batch, target_batch, max_length, device = "cpu"):
    
    # Setup
    model.train()
    model.to(device)
    optimizer.zero_grad()

    # Pad target batch for teacher-forcing
    padded_target_batch = pad_sequence_list_to_max(target_batch, max_length = max_length, pad_value = EOS_index).to(device)

    # Forward pass
    batched_outputs, batched_output_logits = model(embeddings = input_batch, labels = padded_target_batch, use_teacher_forcing = True, max_length = max_length, device = device)

    # Clean and pad output to max_length
    masked_logits, seq_lengths = switch_out_post_eos(batched_output_logits, batched_outputs)
    padded_logits = pad_sequence_tensor_to_max(masked_logits, max_length, EOS_index)

    # Calculate max length out of model generation and target labels
    max_target_size = max(t.size(0) for t in target_batch)
    max_output_size = seq_lengths.max().item()
    max_seq_len = max(max_target_size , max_output_size)

    # Print lengths of prediction and output
    print(f"Max target size {max_target_size}")
    print(f"Max_output_size {max_output_size}")
    print(f"Max seq len: {max_seq_len}")

    # Truncate to this max length length between two
    target_label = padded_target_batch[:, :max_seq_len]
    output_logits = padded_logits[:, :max_seq_len, :]

    # Get loss
    total_loss = 0
    for t in range(max_seq_len):
        loss = loss_fn(output_logits[:, t, :], target_label[:, t])
        total_loss += loss

    total_loss.backward()

    clip_value = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()

    print(f"TRAIN LOSS {total_loss.item()}")
    return total_loss.item()

def evaluate(model, loss_fn, input_batch, target_batch, tgt_vocab, max_length, device = "cpu"):
    # Setup
    model.eval()
    model.to(device)
   
    # Pad target batch 
    padded_target_batch = pad_sequence_list_to_max(target_batch, max_length = max_length, pad_value = SOS_index).to(device)

    # Forward pass - teacher forcing False because eval
    batched_outputs, batched_output_logits = model(embeddings = input_batch, labels = padded_target_batch, use_teacher_forcing = False, max_length = max_length, device = device)

    # Clean and pad output to max_length
    masked_logits, seq_lengths = switch_out_post_eos(batched_output_logits, batched_outputs)
    padded_logits = pad_sequence_tensor_to_max(masked_logits, max_length, SOS_index)

    # Calculate max length out of model generation and target labels
    max_target_size = max(t.size(0) for t in target_batch)
    max_output_size = seq_lengths.max().item()
    max_seq_len = max(max_target_size , max_output_size)

    # Print lengths of prediction and output
    print(f"Max target size {max_target_size}")
    print(f"Max_output_size {max_output_size}")
    print(f"Max seq len: {max_seq_len}")

    # Truncate to this max length
    target_label = padded_target_batch[:, :max_seq_len]
    output_logits = padded_logits[:, :max_seq_len, :]

    # Get loss
    total_loss = 0
    for t in range(max_seq_len):
        loss = loss_fn(output_logits[:, t, :], target_label[:, t])
        total_loss += loss

    # Get actual predictions and actual label tokens
    preds = torch.argmax(output_logits, dim = 2).tolist()
    pred_sentences = [ids_to_sentence(pred, tgt_vocab) for pred in preds]
    target_sentences = [ids_to_sentence(target, tgt_vocab) for target in target_batch]
    print(f"Target sentence : {target_sentences[:1]}")
    print(f"Pred sentences : {pred_sentences[:1]}")

    # Calculate bleu scores across sentences
    scores = [sentence_bleu([target_sent], pred_sent) for (target_sent, pred_sent) in zip(target_sentences, pred_sentences)]

    # Return avg bleu score and dev loss
    return sum(scores) / len(scores), total_loss.item()
    
    
# logging.info(f"Batched outputs shape: {batched_output_probs.shape}")
    # logging.info(f"Batched output probs shape: {batched_output_probs.shape}")
    # print(f"Target batch shape: {target_batch.shape}")
    # logging.info(f"Cleaned probs shape : {cleaned_probs.shape}")
    # logging.info(f"Target batch : {target_batch.shape}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=768, type=int, help = "hidden size of encoder/decoder, also word vector size")
    parser.add_argument('--n_iters', default=1000, type=int, help = "total number of batches to train on")
    parser.add_argument('--tgt_vcb', help = "target vocabulary json")
    parser.add_argument('--print_every', default=10, type=int, help = "print loss info every this many training examples")
    parser.add_argument('--checkpoint_every', default=10, type=int, help = "write out checkpoint every this many batches")
    parser.add_argument('--batch_size', default=16, type=int, help = "batch_size")
    parser.add_argument('--initial_lr', default=0.001, type=float, help = "initial learning rate")
    parser.add_argument("--max_length", default = 60, nargs="?", help = "max number of tokens in generation")
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
    
    train_pairs = get_pairs(args.train_file, max_size=10000)
    dev_pairs = get_pairs(args.dev_file, max_size = 100000)
    # silver_pairs = get_pairs(args.silver_file)
    # test_pairs = get_pairs(args.test_file)

    if args.load_checkpoint:
        state = torch.load(args.load_checkpoint)9