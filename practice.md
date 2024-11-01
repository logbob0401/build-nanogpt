## GPU 现存利用率不高，只有26%（对h800）
### 发现！ 
v100上没有bf16只能用fp16，而且mem小只能用小batch 16 （wordsize 8,T 1024,grad_accum_steps 4），结果导致训练不稳定;
我怀疑fp16更直接有问题。
-->用h800 batch 16+ bf16 没问题
-->不用量化直接慢死：
step    56 | loss: 8.319144 | lr 4.7832e-05 | norm: 6.4308 | dt: 4875.88ms | tok/sec: 107526.94
step    57 | loss: 8.266062 | lr 4.8671e-05 | norm: 6.4732 | dt: 4865.72ms | tok/sec: 107751.38
tok降到10k/s, 8 v100 32g,正常可以到435k/s
-->用compile 会00M 
-->compile false, warmup_steps+2000,前期还行，lr到了1e4后再次变得不稳定,当时norm还是在7-20,看起来某些batch very unlucky
QQQ: how about adjust lr based on norm?
-->compile false, warmup_steps+2000，max_lr = 1e-4 is 1/6 of original: ==> 到了1400多步又出现loss变大，这时lr在6e5，norm8-12
--> compile false, warmup_steps  ,increase gradiant accumulate steps * 4 ==> still not working,loss surge to 18+
--> compile false, warmup_steps 751 ,increase gradiant accumulate steps * 2, lr=lr/norm ==>  still not working, loss surge after 1400 step
--> compile false, warmup_steps 751 ,norm>2 then clip gradient to 0.5; lr=lr/norm ==>  still not working, loss donot reduce after 1400 step
--> frustrate, so let me try back to no cast： no problem
--> GradScaler,在敏感层使用 FP32 计算(laynorm,softmax,gelu),调整学习率到1/6: 刚只是在laynorm+gelu用了fp32，稳定性已经有明显改善,但是在3000+ step之后loss不再降低；这次softmax部分也用上。==>还是不行，loss到了3.5就不降低了,update:17249 train 3.324421, 降低的比fp32和bf16都慢，但还是有效果的；随着更多地方用fp32，tok/sec降低了约一半: 235k/s。看起来可以优化，但是需要更多训练时间才可以降低loss更多；==> 整体认为还是成功的。rec_202411011055_basically_works


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


