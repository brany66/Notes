#!/bin/bash
#
# This script  is use do some preproccess for parallel corpus, and then used in mgiza , moses, and so on.
#
# It contains four part
#
# First is "tokenizer", we use it to insert a blankspace between words and punctuation
#
# Second is "train-truecaser", is extract some statistical information about corpus
#
# Third is "truecase", reduce data sparse
#
# Forth is "clean-corpus-n", is to delete some empty and too long lines
#
# Author: YWJ
# Date： 2016-12-22
# Copyright (c) 2016 NJU PASA Lab All rights reserved.
# 

#MOSES=/home/wmt/mosesdecoder/scripts
MOSES=/home/wmt/moses/scripts
# This is your mgizapp install directory
if [ $# -lt 4 ]; then
    echo "OK, this is simple" 1>&2
    echo "run " $0 " PRE src tgt rootDir, MAXLEN is the max length sentence" 1>&2
    echo "first make sure this script cand find source and target file, and \${QMT_HOME} is in bashrc" 1>&2
    exit
fi

PRE=$1
SRC=$2
TGT=$3
ROOT=$4
MAXLEN=$5

mkdir -p $ROOT/workspace
 
echo "Do tokenizer" 1>&2
${MOSES}/tokenizer/tokenizer.perl -l ${SRC} < ${ROOT}/$PRE.${SRC} > ${ROOT}/workspace/${PRE}.${SRC}.tok.${SRC}

${MOSES}/tokenizer/tokenizer.perl -l ${TGT} < ${ROOT}/$PRE.${TGT} > ${ROOT}/workspace/${PRE}.${TGT}.tok.${TGT}
echo "Done tokenizer" 1>&2

echo "Do train-truecaser" 1>&2
${MOSES}/recaser/train-truecaser.perl --model ${ROOT}/workspace/${PRE}.model.${SRC} --corpus ${ROOT}/workspace/${PRE}.${SRC}.tok.${SRC}

${MOSES}/recaser/train-truecaser.perl --model ${ROOT}/workspace/${PRE}.model.${TGT} --corpus ${ROOT}/workspace/${PRE}.${TGT}.tok.${TGT}

echo "Done train-truecaser" 1>&2

echo "Do truecase" 1>&2
${MOSES}/recaser/truecase.perl --model ${ROOT}/workspace/${PRE}.model.${SRC} < ${ROOT}/workspace/${PRE}.${SRC}.tok.${SRC} > ${ROOT}/workspace/${PRE}.${SRC}-${TGT}.true.${SRC}

${MOSES}/recaser/truecase.perl --model ${ROOT}/workspace/${PRE}.model.${TGT} < ${ROOT}/workspace/${PRE}.${TGT}.tok.${TGT} > ${ROOT}/workspace/${PRE}.${SRC}-${TGT}.true.${TGT}

echo "Done truecase" 1>&2

# n is your max length
echo "Do clean-corpus-n" 1>&2
${MOSES}/training/clean-corpus-n.perl ${ROOT}/workspace/${PRE}.${SRC}-${TGT}.true ${SRC} ${TGT} ${ROOT}/${PRE}.${SRC}-${TGT}.clean 1 ${MAXLEN}
echo "Done clean-corpus-n" 1>&2

#rm -rf ${ROOT}/${PRE}.${SRC}-${TGT}.clean.${SRC}
#mv res.txt ${ROOT}/${PRE}.${SRC}-${TGT}.clean.${SRC}

#rm -rf ${ROOT}/workspace 

