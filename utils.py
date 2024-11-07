import torch
import logging
import os
from datetime import  datetime
# -----------------------------------------------------------------------------
def get_supported_dtype(device):
    # 检测CUDA是否可用
    if not torch.cuda.is_available():
        
        return torch.float32

    # 获取当前设备（GPU）,in ddp this should be 'cuda:{ddp_local_rank}'
    if torch.cuda.get_device_capability(device) >= (8, 0):
        # 大部分支持BFloat16的GPU属于8.0或更高版本架构
        if torch.cuda.is_bf16_supported():
            
            return torch.bfloat16
        else:
            
            return torch.float16
    else:
        
        return torch.float16


def define_logger(master_process,log_dir):
    logger = logging.getLogger("GPTTraining")
    # 配置日志记录器
    logger.setLevel(logging.DEBUG if master_process else logging.CRITICAL)  # 非主进程设为 CRITICAL，不输出内容

    
    os.makedirs(log_dir, exist_ok=True)
    # Logging setup
    current_datetime = datetime.now().strftime("%m%d%H%M")
    rec_file = os.path.join(log_dir, f"rec_{current_datetime}.txt")
    # 设置第一个文件处理器并输出到控制台
    file_handler_1 = logging.FileHandler(rec_file)
    file_handler_1.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置简洁的日志格式（只输出消息内容）
    simple_format = logging.Formatter("%(message)s")
    file_handler_1.setFormatter(simple_format)
    console_handler.setFormatter(simple_format)

    # 添加第一个处理器到日志记录器
    logger.addHandler(file_handler_1)
    logger.addHandler(console_handler)
    return logger
