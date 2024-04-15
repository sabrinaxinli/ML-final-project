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
    ("TGT_VCB", "", ""),
    ("EMB_FILE", "", ""),
    ("PROPORTIONS", "", [0.6, 0.2, 0.1, 0.1]), # train, val, silver, test
    ("DATA_SPLITS", "", [""])
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
            action="python scripts/build_data.py --src_lang ${SRC_LANG} --tgt_lang ${TGT_LANG} --src_bitext ${SRC_BITEXT} --tgt_bitext ${TGT_BITEXT} --max_samples ${MAX_SAMPLES} --vcbs ${VOCABS} --parallel_output ${PARALLEL} --emb_output ${EMB_OUT} --batch_size ${BATCH_SIZE}",
        ),
        "SplitData" : Builder(
            action="python scripts/split_data.py --file1_path ${FILE1} --file2_path ${FILE2} --output_paths ${OUTPUT} --proportions ${PROPORTIONS}"
        )
    }
)

# Get data
if env.get("VOCABS", None) and env.get("EMB_FILE", None) and env.get("PARALLEL", None):
    vocabs = env.File(env["VOCABS"])
    emb_file = env.File(env["EMB_FILE"])
    parallel_file = env.File(env["PARALLEL"])
else:
    vocabs, parallel, emb_out = env.BuildData(SRC_LANG = "eng",
                                      TGT_LANG = "de",
                                      SRC_BITEXT = "./multitarget-ted/en-de/raw/ted_train_en-de.raw.en",
                                      TGT_BITEXT = "./multitarget-ted/en-de/raw/ted_train_en-de.raw.de",
                                      MAX_SAMPLES = 100000,
                                      VOCABS = "./work/eng-de.vcb",
                                      PARALLEL= "./work/eng-de-parallel.jsonlines",
                                      EMB_OUT = "./work/eng-de-embedded.jsonlines",
                                      BATCH_SIZE=64)
    split_outputs = env.SplitData(FILE1 = parallel,
                                  FILE2 = emb_out,
                                  OUTPUT = ["./work/train/en-de-parallel-train.jsonlines", "./work/train/en-de-embedded-train.jsonlines",
                                            "./work/dev/en-de-parallel-train.jsonlines", "./work/dev/en-de-embedded-train.jsonlines",
                                            "./work/silver/en-de-parallel-train.jsonlines", "./work/silver/en-de-embedded-train.jsonlines",
                                            "./work/test/en-de-parallel-train.jsonlines", "./work/test/en-de-embedded-train.jsonlines"],
                                  PROPORTIONS = [0.6, 0.1, 0.2, 0.1]
                                  )
    



