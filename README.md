# PyTorch implementation of Jan Chorowski, Jan 2019 paper"

This is a PyTorch implementation of https://arxiv.org/abs/1901.08810.

[Under Construction]

## Update April 14, 2019

Began training on Librispeech dev (http://www.openslr.org/resources/12/dev-clean.tar.gz),
see dat/example\_train.log

# TODO
1. VAE and VQVAE versions of the bottleneck / training objectives
2. Inference mode
 
# Example training setup

```sh
code_dir=/path/to/ae-wavenet
run_dir=/path/to/my_runs

# Get the data
cd $run_dir
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar zxvf dev-clean.tar.gz
$code_dir/scripts/librispeech_to_rdb.sh LibriSpeech/dev-clean > librispeech.dev-clean.rdb 

# Train
cd $code_dir 
python train.py new -af par/arch.basic.json -tf par/train.basic.json -nb 4 -si 10 \
  -rws 100000 -fpu 1.0 $run_dir/model%.ckpt $run_dir/librispeech.dev-clean.10.r1.rdb
```

