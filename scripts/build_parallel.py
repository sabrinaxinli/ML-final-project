import random
import json
import re

import unicodedata
import argparse

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1_path", default = "./multitarget-ted/en-de/raw/ted_train_en-de.raw.en", help = "File 1 path")
    parser.add_argument("--file2_path", default = "./multitarget-ted/en-de/raw/ted_train_en-de.raw.de", help = "file 2 path")
    parser.add_argument("--output1", default = "eng-de-train.jsonlines", help = "Train")
    parser.add_argument("--output2", default = "eng-de-dev.jsonlines", help = "Dev")
    parser.add_argument("--output3", default = "eng-de-test.jsonlines", help = "Test")
    parser.add_argument("--max_length", type = int)
    parser.add_argument("--train_size", type = int, help = "number of train")
    parser.add_argument("--dev_size", type = int, help = "number of dev")
    parser.add_argument("--test_size", type = int, help = "number of test")
    args, rest = parser.parse_known_args()

    #write the split and shuffled lines to new files
    paired_lines = []
    with open(args.file1_path, "r", encoding="utf-8") as file1, open(args.file2_path, "r", encoding="utf-8") as file2:
        for (line1, line2) in zip(file1, file2):
            if len(line1.split()) < args.max_length and len(line2.split()) < args.max_length:
                paired_lines.append((line1.strip(), line2.strip()))

    #shuffle the pairs
    random.shuffle(paired_lines)

    dev_size = args.train_size + args.dev_size
    test_size = args.train_size + args.dev_size + args.test_size

    train = paired_lines[:args.train_size]
    dev = paired_lines[args.train_size : dev_size]
    test = paired_lines[dev_size : test_size]

    with open(args.output1, 'w', encoding="utf-8") as out1:
        for paired in train:
            out1.write(json.dumps((paired)) + "\n")

    with open(args.output2, 'w', encoding="utf-8") as out2:
        for paired in dev:
            out2.write(json.dumps((paired)) + "\n")

    with open(args.output3, 'w', encoding="utf-8") as out3:
        for paired in test:
            out3.write(json.dumps((paired)) + "\n")