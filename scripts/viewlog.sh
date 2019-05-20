#!/bin/bash

if [ $# -eq 0 ]
then
	echo 'Usage: '
	echo "$(basename $0) file.log [M|S]"
	exit 1
fi

file=$1
mode=$2
tab=$(echo -en '\t')
grep -E "^$mode$tab" $file | column -t -s "$tab"

