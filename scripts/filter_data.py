import random
import argparse
import gzip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1_path", help = "File 1 path")
    parser.add_argument("--file2_path", help = "file 2 path")
    parser.add_argument("--output1", help = "Bitext1")
    parser.add_argument("--output2", help = "Bitext2")
    parser.add_argument("--max_length", type = int, help = "max length of sentence")
    parser.add_argument("--max_docs", type = int, help = "max number of docs")
    args, rest = parser.parse_known_args()

    print(args.file1_path)
    paired_lines = []
    with open(args.file1_path, "r", encoding="utf-8") as file1, open(args.file2_path, "r", encoding="utf-8") as file2:
        for (line1, line2) in zip(file1, file2):
            if len(line1.split()) < args.max_length and len(line2.split()) < args.max_length:
                paired_lines.append((line1.strip(), line2.strip()))

    #shuffle the pairs
    random.shuffle(paired_lines)

    paired_lines = paired_lines[:args.max_docs]

    #write the split and shuffled lines to new files
    with open(args.output1, 'w', encoding="utf-8") as out1, open(args.output2, 'w', encoding="utf-8") as out2:
        for line1, line2 in paired_lines:
            out1.write(f"{line1}\n")
            out2.write(f"{line2}\n")