# Video Model Benchmark

## Environments

* CUDA Toolkit 10.1
* CuDNN 7.6.2

Install dependencies for [SlowFast](https://github.com/facebookresearch/SlowFast). Then create link by

```
ln -s SlowFast/slowfast ./slowfast
```

## Getting started

Profile by running

```
python main.py --model i3d
```