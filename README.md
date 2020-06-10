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

## Example output

```
Profile model i3d

Params: 28,043,472
GPU Mem: 0.10470056533813477 MB
FLOPs: 56.814501888 GFLOPs

Module                | Self CPU total | CPU total | CUDA total | Occurrences
----------------------|----------------|-----------|------------|------------
ResNet                |                |           |            |            
├── s1                |                |           |            |            
│├── pathway0_stem    |                |           |            |            
││├── conv            |        1.448ms |   5.687ms |  102.623ms |           1
││├── bn              |        1.314ms |   3.763ms |   10.265ms |           1
││├── relu            |       48.309us |  48.309us |    1.073ms |           1
││└── pool_layer      |        1.579ms |   3.144ms |    4.938ms |           1
│ ...
└── head              |                |           |            |            
 ├── pathway0_avgpool |       67.693us |  67.693us |  961.536us |           1
 ├── dropout          |       71.013us | 127.499us |  131.072us |           1
 ├── projection       |      250.353us | 399.099us |  410.624us |           1
 └── act              |        0.000us |   0.000us |    0.000us |           0
```

View trace in [chrome://tracing](chrome://tracing) or [edge://tracing](edge://tracing). 

![](https://i.loli.net/2020/06/10/lky2e4QJvn3YGCR.png)