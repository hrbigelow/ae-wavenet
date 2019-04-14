#!/bin/bash

if [ $# -eq 0 ]
then
    echo 'Usage:'
    echo "$(basename $0) /path/to/LibriSpeech/{dev-clean,test-clean} > librispeech.rdb"
    exit 1
fi

datadir=$(readlink -m $1)
for dirpath in $(find $datadir -maxdepth 1 -type d -regex '.+[0-9]+')
do
    speaker_id=$(basename $dirpath)
    for filepath in $(find $dirpath -type f -name '*.flac')
    do
        echo -en "$speaker_id\t$filepath\n"
    done
done

