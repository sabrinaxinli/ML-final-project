import gzip
with gzip.open("./train/en-de-embedded_train_saved.jsonlines", "r") as input, gzip.open("output.jsonlines.gz", "w") as output:
    i = 0
    for line in input:
        output.write(line)
        if i > 9999:
            break
        i+=1
