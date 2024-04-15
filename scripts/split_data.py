import random
import argparse
# def shuffle_and_split_files(file1_path, file2_path, output_paths, proportions):
    # Read the contents of both files
   

# the paths to the original files and the output files
# file1_path = '/Users/jiasenjing/Desktop/code/ml/projcet/ted_train_en-de.raw.de'
# file2_path = '/Users/jiasenjing/Desktop/code/ml/projcet/ted_train_en-de.raw.en'
# output_paths = [
#     ('/Users/jiasenjing/Desktop/code/ml/projcet/train.de', '/Users/jiasenjing/Desktop/code/ml/projcet/train.en'), # output for train
#     ('/Users/jiasenjing/Desktop/code/ml/projcet/val.de', '/Users/jiasenjing/Desktop/code/ml/projcet/val.en'), # output for validation
#     ('/Users/jiasenjing/Desktop/code/ml/projcet/silver.de', '/Users/jiasenjing/Desktop/code/ml/projcet/silver.en'), # output for silver
#     ('/Users/jiasenjing/Desktop/code/ml/projcet/test.de', '/Users/jiasenjing/Desktop/code/ml/projcet/test.en') # output for test
# ]

# Define the proportions for train, val, silver, and test in that order
# proportions = [0.6, 0.2, 0.1, 0.1]  

# shuffle_and_split_files(file1_path, file2_path, output_paths, proportions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1_path", help = "File 1 path")
    parser.add_argument("--file2_path", help = "file 2 path")
    parser.add_argument("--output_paths", nargs = "+", help = "Output paths as list (assumed to be paired)")
    parser.add_argument("--proportions", type = int, nargs = "+", help = "list of proportions corresponding to split sizes")
    args, rest = parser.parse_known_args()

    if len(args.proportions) != (len(args.output_paths) // 2):
        raise ValueError("Output size and proportion nums do not match")

    with open(args.file1_path, "r", encoding="utf-8") as file1:
        lines1 = file1.readlines()
    with open(args.file2_path, "r", encoding="utf-8") as file2:
        lines2 = file2.readlines()

    #zip the lines from both files
    paired_lines = list(zip(lines1, lines2))
    
    #shuffle the pairs
    random.shuffle(paired_lines)

    #calculate the split indices
    total_lines = len(paired_lines)
    split_points = [int(total_lines * sum(args.proportions[:i+1])) for i in range(len(args.proportions))]
    
    #split the pairs into specified proportions
    split_data = [paired_lines[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]
    split_data.insert(0, paired_lines[:split_points[0]])

    #write the split and shuffled lines to new files
    for i, data in enumerate(split_data):
        shuffled_lines1, shuffled_lines2 = zip(*data)
        with open(args.output_paths[i], 'w', encoding='utf-8') as out1:
            out1.writelines(shuffled_lines1)
        with open(args.output_paths[i+1], 'w', encoding='utf-8') as out2:
            out2.writelines(shuffled_lines2)