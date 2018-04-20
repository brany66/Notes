# 语料准备阶段
1. 若需要分词，use jieba 分词
2. 全角转半角：full2half.py
3. four step: tokenizer train-truecaser, truecase, clean-corpus-n
4. if need lowercase :
#  cat ~/data/UNPC/en-zh/clean/UNPC.en-zh.zh | ./lowercase.perl > ~/data/UNPC/en-zh/clean/UNPC.en-zh.lower.zh

# language model
# https://kheafield.com/code/kenlm/estimation/
-o: Required. Order of LM
-S : Recommmended. Memory to use, 80%
-T: temp file location

/home/wmt/moses/bin/lmplz -o 3 -S 60% -T tmp/ < /home/wmt/data/UNPC/en-zh/clean/UNPC.en-zh.lower.1w.en > /home/wmt/workDir/LM/UNPC.en-zh.lower.1w.arpa.en
/home/wmt/moses/bin/build_binary /home/wmt/workDir/LM/UNPC.en-zh.lower.1w.arpa.en /home/wmt/workDir/LM/UNPC.en-zh.lower.1w.blm.en

# echo "is this an English sentence ?" | ~/moses/bin/query ~/workDir/LM/UNPC.en-zh.lower.1w.blm.en


# 词对齐模型

# mgizapp.sh
# if have test.zh test.en workingDir: training
# test en zh training 0


# translation model


# Tuning

# Testing

 
