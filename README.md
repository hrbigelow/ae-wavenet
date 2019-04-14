# PyTorch implementation of Jan Chorowski, Jan 2019 paper"

This is a PyTorch implementation of https://arxiv.org/abs/1901.08810.

[Under Construction]

[Update April 14, 2019]

Began training on Librispeech dev (http://www.openslr.org/resources/12/dev-clean.tar.gz),
see dat/example\_train.log

[Example setup]

```sh
$ mkdir my\_runs && cd my\_runs
$ wget http://www.openslr.org/resources/12/dev-clean.tar.gz
$ tar zxvf dev-clean.tar.gz
$ cd /path/to/ae-wavenet
$ ./scripts/librispeech\_to\_rdb.sh LibriSpeech/dev-clean \
  > /path/to/my\_runs/librispeech.dev-clean.rdb 
$ python train.py new -af par/arch.basic.json -tf par/train.basic.json -nb 4 -si 10 \
  -rws 100000 -fpu 1.0 /path/to/my\_runs/model%.ckpt \
  /path/to/my\_runs/librispeech.dev-clean.10.r1.rdb
```

