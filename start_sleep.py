import torch

import time

print('PyTorch 版本:', torch.__version__)
print('CUDA 是否可用:', torch.cuda.is_available())
print('CUDA 版本:', torch.version.cuda)
print('CUDA 设备数量:', torch.cuda.device_count())


def sleep_forever():
    try:
        while True:
            time.sleep(1)  # 每次休眠1秒，防止占用过多的CPU资源
    except KeyboardInterrupt:
        print("Sleep interrupted by user")

# 调用这个方法
sleep_forever()