## GPU 现存利用率不高，只有26%（对h800）
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
memory.used [MiB], memory.total [MiB]
22001 MiB, 81559 MiB
22001 MiB, 81559 MiB
### 优化空间300%, 
### (Done)尝试加大batch 16->64(good)->128(oom)
### after update batch  to 64, tok/s increase to 565k/s, memory used about 79GB+, use rate increase from 26p to 97p
step   498 | loss: 5.180400 | lr 4.1874e-04 | norm: 1.0096 | dt: 927.33ms | tok/sec: 565373.84
step   499 | loss: 5.164181 | lr 4.1958e-04 | norm: 1.1487 | dt: 928.54ms | tok/sec: 564638.99
validation loss: 5.3183
HellaSwag accuracy: 2391/10042=0.2381
step   500 | loss: 5.209949 | lr 4.2042e-04 | norm: 1.0975 | dt: 25602.18ms | tok/sec: 20478.26
step   501 | loss: 5.164941 | lr 4.2126e-04 | norm: 0.8869 | dt: 929.10ms | tok/sec: 564293.86
    和context
### 尝试更大的模型

# 数据load可能有太耗时问题，尝试用数据预取（Data Prefetching）或异步数据加载（Asynchronous Data Loading）优化
注意load新batch的和新shard的耗时
step 10750 | loss: 3.170763 | lr 2.9059e-04 | norm: 0.3097 | dt: 24613.12ms | tok/sec: 21301.16
......
step 11024 | loss: 3.129706 | lr 2.7811e-04 | norm: 0.2697 | dt: 1014.05ms | tok/sec: 517024.54
step 11025 | loss: 3.133868 | lr 2.7807e-04 | norm: 0.3027 | dt: 1015.43ms | tok/sec: 516321.91
......
step 11058 | loss: 3.122603 | lr 2.7657e-04 | norm: 0.2706 | dt: 1013.66ms | tok/sec: 517223.98
step 11059 | loss: 3.166646 | lr 2.7653e-04 | norm: 0.2663 | dt: 91965.31ms | tok/sec: 5700.93
step 11060 | loss: 3.207780 | lr 2.7648e-04 | norm: 0.2743 | dt: 1014.06ms | tok/sec: 517019.07
......
step 12585 | loss: 3.120876 | lr 2.1001e-04 | norm: 0.2795 | dt: 185822.45ms | tok/sec: 2821.45
step 12586 | loss: 3.135432 | lr 2.0997e-04 | norm: 0.2679 | dt: 1013.50ms | tok/sec: 517302.58
## 大约的优化幅度 10%

# (done) data shuffle，提高训练平滑性

# 解决compile 问题 
https://github.com/karpathy/build-nanogpt/pull/62/files
https://github.com/karpathy/build-nanogpt/pull/73

# (done) loss calculation to imporve accuracy by reducing floating-point errors

# (done) Ensure Consistency Between GPTConfig.block_size and Sequence Length T
https://github.com/karpathy/build-nanogpt/pull/72/files

# (done)improve log, add more info


# add load check point and resume part
https://github.com/karpathy/build-nanogpt/pull/83/files


