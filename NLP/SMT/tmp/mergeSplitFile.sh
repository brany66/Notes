#!/bin/bash
list_allDir() {
    for file2 in `ls -a $1`
    do
	if [ x"$file2" != x"." -a x$"file2" != x".." ]; then
           if [ -f "$1/$file2" ]; then
	      cat "$1/$file2" >> $2
	   fi
	fi
    done
}
list_allDir /home/experiment/ywj/data/UNPC/UNPC.align /home/experiment/ywj/data/UNPC/UNPC.merged
#cat /home/experiment/ywj/data/UNPC/UNPC.merged | sed -e '/ALIGN_ERR/d' | sed 's/{##}/\t/g' > /home/experiment/ywj/data/UNPC/UNPC.clean.merged

#rm -rf /home/experiment/ywj/data/UNPC/UNPC.merged

 
#list_allDir /home/experiment/ywj/data/UNPC/UNPC-Syntax /home/experiment/ywj/data/UNPC/UNPC.mergedAll
#cat /home/experiment/ywj/data/UNPC/UNPC.mergedAll | sed 's/|||/\t/g' > /home/experiment/ywj/data/UNPC/mergedAll
