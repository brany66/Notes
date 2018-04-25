#!/bin/bash
#
# This file is part of word alignment, It will do three part

# First is "plaint2snt", it generate two kind of  format files : *.vcb and *.snt

# Second is "sntcooc", it generate one kind of format file "*cooc"

# Third is "mkcls", it will generate one kind of format file ".classes"
#
# Author: YWJ
# Date： 2016-12-21
# Copyright (c) 2016 NJU PASA Lab All rights reserved.
# 

MGIZA=${QMT_HOME}/bin/mgiza
if [ $# -lt 4 ]; then
    echo "OK, this is simple, needs four parameters" 1>&2
    echo "PRE is prefix of filename, SRC is source , TGT is target, root is your workspace, " 1>&2
    echo "run " $0 " PRE src tgt rootDir" 1>&2
    echo "first make sure this script cand find source and target file, and \${QMT_HOME} is in bashrc" 1>&2
    exit
fi

PRE=$1
SRC=$2
TGT=$3
ROOT=$4
NUM=$5

mkdir -p $ROOT/giza-inverse.${NUM}
mkdir -p $ROOT/giza.${NUM}
mkdir -p $ROOT/prepared.${NUM}

echo "Do First part : plain2snt" 1>&2
${QMT_HOME}/bin/plain2snt $ROOT/$PRE.$SRC $ROOT/$PRE.$TGT

echo "Done First part" 1>&2

echo "Do Second part : snt2cooc" 1>&2
${QMT_HOME}/bin/snt2cooc $ROOT/giza.${NUM}/$TGT-$SRC.cooc $ROOT/$PRE.$SRC.vcb $ROOT/$PRE.$TGT.vcb $ROOT/$PRE.$TGT\_$PRE.$SRC.snt
${QMT_HOME}/bin/snt2cooc $ROOT/giza-inverse.${NUM}/$SRC-$TGT.cooc $ROOT/$PRE.$TGT.vcb $ROOT/$PRE.$SRC.vcb $ROOT/$PRE.$SRC\_$PRE.$TGT.snt 
echo "Done Second part!" 1>&2

echo "Do Third part: mkcls" 1>&2
${QMT_HOME}/bin/mkcls -c50 -n5 -p$ROOT/$PRE.$SRC -V$ROOT/$PRE.$SRC.vcb.classes opt
${QMT_HOME}/bin/mkcls -c50 -n5 -p$ROOT/$PRE.$TGT -V$ROOT/$PRE.$TGT.vcb.classes opt
echo "Done Third" 1>&2

#
# This file is use mgiza++ do word alignment, It contains two part
#
# First is "mgiza", it generate two direction word alignment information, in this part you can set training configuration
#
# Second is script "merge_alignment.py", it will compress files generate in first part
#

echo "Do First part : mgiza" 1>&2

${QMT_HOME}/bin/mgiza -ncpu 8 -c $ROOT/$PRE.${SRC}_$PRE.$TGT.snt -o $ROOT/giza-inverse.${NUM}/$SRC-${TGT} \
 -s $ROOT/$PRE.$SRC.vcb -t $ROOT/$PRE.$TGT.vcb -coocurrence $ROOT/giza-inverse.${NUM}/$SRC-${TGT}.cooc \
 -m1 7 -m2 0 -mh 8 - m3 3 -m4 3
 #-restart 11 -previoust $ROOT/giza-inverse.${NUM}/$SRC-$TGT.t3.final \
 #-previousa $ROOT/giza-inverse.${NUM}/$SRC-$TGT.a3.final -previousd $ROOT/giza-inverse.${NUM}/$SRC-$TGT.d3.final \
 #-previousn $ROOT/giza-inverse.${NUM}/$SRC-$TGT.n3.final -previousd4 $ROOT/giza-inverse.${NUM}/$SRC-$TGT.d4.final \
 #-previousd42 $ROOT/giza-inverse.${NUM}/$SRC-$TGT.D4.final -m3 0 -m4 1

${QMT_HOME}/bin/mgiza -ncpu 8 -c $ROOT/$PRE.${TGT}_$PRE.$SRC.snt -o $ROOT/giza.${NUM}/$TGT-${SRC} \
 -s $ROOT/$PRE.$TGT.vcb -t $ROOT/$PRE.$SRC.vcb -coocurrence $ROOT/giza.${NUM}/$TGT-${SRC}.cooc \
 -m1 7 -m2 0 -mh 8 - m3 3 -m4 3
 #-restart 11 -previoust $ROOT/giza-inverse.${NUM}/$SRC-$TGT.t3.final \
 #-previousa $ROOT/giza-inverse.${NUM}/$SRC-$TGT.a3.final -previousd $ROOT/giza-inverse.${NUM}/$SRC-$TGT.d3.final \
 #-previousn $ROOT/giza-inverse.${NUM}/$SRC-$TGT.n3.final -previousd4 $ROOT/giza-inverse.${NUM}/$SRC-$TGT.d4.final \
 #-previousd42 $ROOT/giza-inverse.${NUM}/$SRC-$TGT.D4.final -m3 0 -m4 1

echo "Done First part" 1>&2

echo "Do compression, merge_alignment.py ." 1>&2
${QMT_HOME}/scripts/merge_alignment.py $ROOT/giza-inverse.${NUM}/$SRC-${TGT}.A3.final.part* | gzip -c > $ROOT/giza-inverse.${NUM}/$SRC-$TGT.A3.final.gz

${QMT_HOME}/scripts/merge_alignment.py $ROOT/giza.${NUM}/$TGT-${SRC}.A3.final.part* | gzip -c > $ROOT/giza.${NUM}/$TGT-$SRC.A3.final.gz 

#${QMT_HOME}/scripts/giza2bal.pl -d "gzip -cd $ROOT/giza.${NUM}/$TGT-$SRC.A3.final.gz" -i "gzip -cd $ROOT/giza-inverse.${NUM}/$SRC-$TGT.A3.final.gz" | ${QMT_HOME}/bin/symal -a="grow" -d="yes" -f="yes" -b="yes" > $ROOT/grow-diag-final-and

echo "Done Second part!" 1>&2


# This file is use mgiza++ merge two direction word alignment.
#
# script is symal and giza2bal.py, use method "grow-diag-final-and"
#
#


echo "Do giza2bal.py & symal" 1>&2
#${QMT_HOME}/scripts/merge_alignment.py $ROOT/giza-inverse.${NUM}/$SRC-${TGT}.A3.final.part* | gzip -c > $ROOT/giza-inverse.${NUM}/$SRC-$TGT.A3.final.gz

#${QMT_HOME}/scripts/merge_alignment.py $ROOT/giza.${NUM}/$TGT-${SRC}.A3.final.part* | gzip -c > $ROOT/giza.${NUM}/$TGT-$SRC.A3.final.gz 

${QMT_HOME}/scripts/giza2bal.pl -d "gzip -cd $ROOT/giza.${NUM}/$TGT-$SRC.A3.final.gz" -i "gzip -cd $ROOT/giza-inverse.${NUM}/$SRC-$TGT.A3.final.gz" | ${QMT_HOME}/bin/symal -a="grow" -d="yes" -f="yes" -b="yes" > $ROOT/grow-diag-final-and.${NUM}

echo "Done" 1>&2

cat $ROOT/grow-diag-final-and.${NUM} | sed -e '/ALIGN_ERR/d' |  sed 's/{##}/\t/g' > ${ROOT}/${PRE}.clean.tmp.${NUM}
rm -rf $ROOT/grow-diag-final-and.${NUM}
mv ${ROOT}/${PRE}.clean.tmp.${NUM} $ROOT/grow-diag-final-and.${NUM}

#rm -rf $ROOT/giza-inverse.${NUM}
#rm -rf $ROOT/giza.${NUM}
#rm -rf $ROOT/prepared.${NUM} 
