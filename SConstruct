import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
import steamroller
import jsonlines
# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))

# Variables control various aspects of the experiment.  Note that you have to declare
# any variables you want to use here, with reasonable default values, but when you want
# to change/override the default values, do so in the "custom.py" file (see it for an
# example, changing the number of folds).
vars = Variables("custom.py")
vars.AddVariables(
    ("TGT_VCB", "", "./work/de_saved.vcb"),
    ("EMB_FILE", "", "./work/eng-de-embedded_saved.jsonlines.gz"),
    ("PARALLEL", "", "./work/eng-de-parallel.jsonlines.gzip"),
    ("PROPORTIONS", "", [0.6, 0.2, 0.1, 0.1]), # train, val, silver, test
    ("DATA_SPLITS", "", []),
    ("USE_GRID", "", ""),
    ("MODEL_PATH", "", ""),
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    ENV=os.environ,
    tools=[steamroller.generate],
    
    # Defining a bunch of builders (none of these do anything except "touch" their targets,
    # as you can see in the dummy.py script).  Consider in particular the "TrainModel" builder,
    # which interpolates two variables beyond the standard SOURCES/TARGETS: PARAMETER_VALUE
    # and MODEL_TYPE.  When we invoke the TrainModel builder (see below), we'll need to pass
    # in values for these (note that e.g. the existence of a MODEL_TYPES variable above doesn't
    # automatically populate MODEL_TYPE, we'll do this with for-loops).
    BUILDERS={
        "BuildData" : Builder(
            # action="python scripts/create_data.py --data_path ${SOURCES} --output ${TARGETS} --granularity $SEGMENT_BY_PG",
            action = "python scripts/build_data.py --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} --src_bitext ${SOURCES[0]} --tgt_bitext ${SOURCES[1]} --max_samples ${MAX_SAMPLES} --src_vcb ${TARGETS[0]} --tgt_vcb ${TARGETS[1]} --parallel_output ${TARGETS[2]} --emb_output ${TARGETS[3]} --batch_size ${BATCH_SIZE}",
        ),
        "SplitData" : Builder(
            action = "python scripts/split_data.py --file1_path ${SOURCES[0]} --file2_path ${SOURCES[1]} --output_paths ${TARGETS} --proportions ${PROPORTIONS}"
        ),
        "TrainModel": Builder(
            action = "python scripts/train_model.py --hidden_size ${HIDDEN_SIZE} --n_iters ${N_ITERS} --tgt_vcb ${VCBS} --print_every ${PRINT_EVERY} --checkpoint_every ${CHK_EVERY} --batch_size ${BATCH_SIZE} --initial_lr ${INIT_LR} --train_file ${SOURCES[0]} --dev_file ${SOURCES[1]} --silver_file ${SOURCES[2]} --test_file ${SOURCES[3]} --out_file ${TARGETS[0]} --load_checkpoint ${LOAD_CHK}"
        )
    }
)

# Build and embed data if no path given
if env.get("TGT_VCB", None) and env.get("EMB_FILE", None) and env.get("PARALLEL", None):
    tgt_vocab = env.File(env["TGT_VCB"])
    emb_file = env.File(env["EMB_FILE"])
    parallel_file = env.File(env["PARALLEL"])
else:
    src_vocab, tgt_vocab, parallel_file, emb_file = env.BuildData(source = ["./multitarget-ted/en-de/raw/ted_train_en-de.raw.en", "./multitarget-ted/en-de/raw/ted_train_en-de.raw.de"],
                                              target = ["./work/eng.vcb", "./work/de.vcb", "./work/eng-de-parallel.jsonlines.gzip", "./work/eng-de-embedded.jsonlines.gzip"],
                                              SRC_LANG = "eng",
                                              TGT_LANG = "de",
                                              MAX_SAMPLES = 100000,
                                              BATCH_SIZE=64)

# Build splits
if env.get("DATA_SPLITS", None):
    split_outputs = env["DATA_SPLITS"]
else:
    print("HERE")
    split_source = [parallel_file, emb_file]
    print(f"Split source: {split_source}")
    split_outputs = env.SplitData(source = split_source,
                                target = ["./work/train/en-de-parallel-train.jsonlines", "./work/dev/en-de-parallel-dev.jsonlines", "./work/silver/en-de-parallel-silver.jsonlines", "./work/test/en-de-parallel-test.jsonlines",
                                            "./work/train/en-de-embedded-train.jsonlines", "./work/dev/en-de-embedded-dev.jsonlines", "./work/silver/en-de-embedded-silver.jsonlines", "./work/test/en-de-embedded-test.jsonlines"],
                                PROPORTIONS = [0.6, 0.1, 0.2, 0.1]
                                )

# If no trained model path, train a model
# if env.get("MODEL_PATH", None):
#     model = env["MODEL_PATH"]
# else:
#     training_splits = env["DATA_SPLITS"] if env.get("DATA_SPLITS", None) else [split_outputs[4:]]
#     result = env.TrainModel(source = training_splits,
#                             target = ["./work/result/results.txt"],
#                             HIDDEN_SIZE = 768,
#                             N_ITERS = 1000,
#                             VCBS = tgt_vocab,
#                             PRINT_EVERY = 5,
#                             CHK_EVERY = 250,
#                             BATCH_SIZE = 16,
#                             INIT_LR = 0.001,
#                             LOAD_CHK = "")

   #  ["./work/train/en-de-parallel-train.jsonlines", "./work/train/en-de-embedded-train.jsonlines",
   #  "./work/dev/en-de-parallel-train.jsonlines", "./work/dev/en-de-embedded-train.jsonlines",
   #  "./work/silver/en-de-parallel-train.jsonlines", "./work/silver/en-de-embedded-train.jsonlines",
   #  "./work/test/en-de-parallel-train.jsonlines", "./work/test/en-de-embedded-train.jsonlines"]
    
    



