
import argparse
import logging
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import json
from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1

MAX_TOKS = 512

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code, file=None):
        if file:
            with open(file, "r") as fp:
                vocab_dict = json.load(fp)
                self.lang_code = vocab_dict["lang_code"]
                self.word2index = vocab_dict["word2index"]
                self.word2count = vocab_dict["word2count"]
                self.index2word = vocab_dict["index2word"]
                self.n_words = vocab_dict["n_words"]
        else:
                self.lang_code = lang_code
                self.word2index = {}
                self.word2count = {}
                self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
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


######################################################################

def add_lines(input_file, vcb, encoding):
    lines = []
    with open(input_file, "r", encoding=encoding) as input:
        for line in input:
            vcb.add_sentence(line)
            lines.append(line)
    return lines, vcb

######################################################################

def make_vocab(lang_code, input_file, encoding):
    vocab = Vocab(lang_code)

    lines, vcb = add_lines(input_file, vocab, encoding)

    logging.info('%s (tgt) vocab size: %s', vocab.lang_code, vocab.n_words)

    return lines, vcb

######################################################################

# each sentence is converted to a tensor of dim [sequence_length, 1]
# we will want to change this to [sequence_length, batch_size] by stacking along dim = 1
def tensor_from_sentence(vocab, sentence, device="cpu"):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1) # [1, seq_len] --> [batch_size, seq_len]

######################################################################

def get_batches(sentences, batch_size):
    curr_batch = []
    batches = []

    for sent in sentences:
        if len(curr_batch) < batch_size:
            curr_batch.append(sent)
        else:
            batches.append(curr_batch)
            curr_batch = [sent]
    
    if curr_batch:
        batches.append(curr_batch)

    return batches

######################################################################

def write_out_embeddings(model, tokenizer, src_sentences, tgt_sentences, output_path, batch_size, device="cpu"):
    model.to(device)
    sentence_pairs = list(zip(src_sentences, tgt_sentences))
    batches = get_batches(sentence_pairs, batch_size)
    
    with open(output_path, "w") as output:
        print(f"Num batches: {len(batches)}")
        for i, batch in enumerate(batches):
            (src_batch, tgt_batch) = zip(*batch)
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
            s_embeddings = torch.split(cls_token_batch, split_size_or_sections=1, dim=0) # list of [1, hidden_size] == [1, 768] for each sequence
            assert len(tgt_batch) == len(s_embeddings)
            
            # Build out data to be format: src_embedding, tgt_ids
            datapoints = list(zip(s_embeddings, tgt_batch))
            for (embedding, tgt_sent) in datapoints:
                output.write(json.dumps((embedding.tolist(), tgt_sent)) + "\n")
            
            print(f"Done with batch {i}")

######################################################################
def id_list_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return indexes

######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", default="eng",
            help='Source (input) language code, e.g. "fr"')
    parser.add_argument("--tgt_lang", default="ger",
            help='Source (input) language code, e.g. "en"'
               
    parser.add_argument("--src_bitext", help="source bitext should have one sentence per line")
    parser.add_argument("--tgt_bitext", help="tgt bitext should have one sentence per line, same length as src_bitext")
    parser.add_argument("--max_samples", type = int, help="Maximum number of datapoints")
    parser.add_argument("--vcbs", help="Source and target vocab json output file")
    parser.add_argument("--parallel_output", help="src sentence ||| tgt sentence")
    parser.add_argument("--emb_output",help='output file for test translations')
    parser.add_argument("--batch_size", type = int, help='output file for test translations')
    parser.add_argument("--encoding", default = "utf-8", help = "encoding type")
    
    args, rest = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)
    torch.cuda.empty_cache()
    
    src_lines, src_vocab = make_vocab(args.src_lang, args.src_bitext, args.encoding)
    tgt_lines, tgt_vocab = make_vocab(args.tgt_lang, args.tgt_bitext, args.encoding)
    
    src_lines = src_lines[:args.max_samples]
    tgt_lines = tgt_lines[:args.max_samples]

    vcbs = {"src": src_vocab.to_dict(), "tgt": tgt_vocab.to_dict()}
    with open(args.vcbs, "w", encoding=args.encoding) as vocab_file:
        json.dump(vcbs, vocab_file)

    with open(args.parallel_output, "w", encoding=args.encoding) as parallel_file:
        for (src_line, tgt_line) in zip(src_lines, tgt_lines):
            parallel_file.write(json.dumps((src_line, tgt_line)) + "\n")

    tgt_id_lists = []
    for tgt_line in tgt_lines:
        tgt_id_lists.append(id_list_from_sentence(tgt_vocab, tgt_line))
    
    assert len(src_lines) == len(tgt_lines)
    assert len(src_lines) == len(tgt_id_lists)

    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    write_out_embeddings(model, tokenizer, src_lines, tgt_id_lists, args.emb_output, args.batch_size, device)


    
    

# def split_lines(input_file):
#     """split a file like:
#     first src sentence|||first tgt sentence
#     second src sentence|||second tgt sentence
#     into a list of things like
#     [("first src sentence", "first tgt sentence"), 
#         ("second src sentence", "second tgt sentence")]
#     """
#     logging.info("Reading lines of %s...", input_file)
#     # Read the file and split into lines
#     lines = open(input_file, encoding='utf-8').read().strip().split('\n')
#     # Split every line into pairs
#     pairs = [l.split('|||') for l in lines]
#     return pairs

# def make_vocabs(src_lang_code, tgt_lang_code, train_file):
#     """ Creates the vocabs for each of the langues based on the training corpus.
#     """
#     src_vocab = Vocab(src_lang_code)
#     tgt_vocab = Vocab(tgt_lang_code)

#     train_pairs = split_lines(train_file)

#     for pair in train_pairs:
#         src_vocab.add_sentence(pair[0])
#         tgt_vocab.add_sentence(pair[1])

#     logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
#     logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

#     return src_vocab, tgt_vocab

# def tensors_from_pair(src_vocab, tgt_vocab, pair):
#     """creates a tensor from a raw sentence pair
#     """
#     input_tensor = tensor_from_sentence(src_vocab, pair[0])
#     target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
#     return input_tensor, target_tensor