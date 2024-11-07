import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples,get_most_likely_row
from datetime import datetime
import gc
import tiktoken
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FlatParameter,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import logging
from functools import partial
import functools
from contextlib import nullcontext
from utils import get_supported_dtype,define_logger
ddp_local_rank = 0
device = "cpu"
# 确定是否为主进程
master_process = (not torch.cuda.is_available() ) or ( int(os.getenv("RANK", "0")) == 0 )

print(f"{master_process=}")

log_dir = "log"

logger=define_logger(master_process,log_dir)

class CausalSelfAttention_MHA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        with torch.cuda.amp.autocast(enabled=False):

            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        #y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

#GQA=GroupedQueryAttention
class CausalSelfAttention_GQA(nn.Module):
    """
    Implements the Grouped Query Attention (GQA) mechanism.
    Optimized using Flash Attention.

    Args:
        config: Configuration object containing model parameters.
            - config.n_embd: The embedding dimension (c).
            - config.n_head: The number of attention heads (h).
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd        # Input feature dimension (c)
        self.num_heads = config.n_head  # Number of attention heads (h)
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.head_dim = self.n_embd // self.num_heads  # Dimension per head (d)

        # Number of key-value heads is hardcoded to half of query heads
        self.num_key_value_heads = self.num_heads // 2  # (kv_h)
        # Number of key-value groups is 2 (since num_heads // num_key_value_heads = 2)
        self.num_key_value_groups = 2

        # Define linear projection layers
        self.q_proj = nn.Linear(self.n_embd, self.n_embd)
        self.k_proj = nn.Linear(self.n_embd, self.n_embd//2)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd//2)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)

    def _repeat_kv(self, key: torch.Tensor, value: torch.Tensor):
        """
        Repeat key and value tensors to match the number of query heads.
        Assumes num_key_value_groups = 2.

        Args:
            key: Key tensor of shape (b, t, kv_h, d)
            value: Value tensor of shape (b, t, kv_h, d)

        Returns:
            key: Repeated key tensor of shape (b, t, h, d)
            value: Repeated value tensor of shape (b, t, h, d)
        """
        if self.num_key_value_groups == 1:
            return key, value

        b, t, kv_h, d = key.size()

        # Expand key tensor without actual data copying
        key = key.unsqueeze(2)                      # (b, t, 1, kv_h, d)
        key = key.expand(-1, -1, 2, -1, -1)         # (b, t, 2, kv_h, d)
        key = key.reshape(b, t, self.num_heads, d)  # (b, t, h, d)

        # Do the same for value tensor
        value = value.unsqueeze(2).expand(-1, -1, 2, -1, -1).reshape(b, t, self.num_heads, d)

        return key, value

    def forward(self, x, attention_mask=None, use_cache=False):
        """
        Forward pass for the GQA layer.

        Args:
            x: Input tensor of shape (b, t, c)
            attention_mask: Optional attention mask tensor.
            use_cache: Whether to use and return cache.

        Returns:
            output: Output tensor of shape (b, t, c)
            past_key_value: Tuple of key and value tensors if use_cache is True
        """
        b, t, c = x.shape  # x: (b, t, c)

        # Compute queries, keys, and values
        q = self.q_proj(x)  # q: (b, t, h * d)
        k = self.k_proj(x)  # k: (b, t, kv_h * d)
        v = self.v_proj(x)  # v: (b, t, kv_h * d)

        # Reshape to separate heads
        q = q.view(b, t, self.num_heads, self.head_dim)           # q: (b, t, h, d)
        k = k.view(b, t, self.num_key_value_heads, self.head_dim) # k: (b, t, kv_h, d)
        v = v.view(b, t, self.num_key_value_heads, self.head_dim) # v: (b, t, kv_h, d)

        # Repeat keys and values to match query heads
        k, v = self._repeat_kv(k, v)  # Now k, v: (b, t, h, d)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # q: (b, h, t, d)
        k = k.transpose(1, 2)  # k: (b, h, t, d)
        v = v.transpose(1, 2)  # v: (b, h, t, d)

        # Use Flash Attention if available
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            #attn_mask=attention_mask,  # Should be of shape (b, 1, t, t) or broadcastable
            #dropout_p=0.0, #self.dropout if self.training else 
            is_causal=True  # Set to True if causal masking is needed
        )  # attn_output: (b, h, t, d)
        if 0:
            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b, h, t, t)
            if attention_mask is not None:
                scores += attention_mask  # Apply attention mask
            attn_weights = F.softmax(scores, dim=-1)  # (b, h, t, t)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)  # (b, h, t, d)

        # Transpose and reshape to merge heads
        attn_output = attn_output.transpose(1, 2).reshape(b, t, c)  # (b, t, c)

        # Final linear projection
        output = self.o_proj(attn_output)  # (b, t, c)

        if use_cache:
            # Return key and value tensors for caching
            past_key_value = (k, v)  # Each of shape (b, h, t, d)
            return output, past_key_value

        return output  #, None

CausalSelfAttention = CausalSelfAttention_GQA
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU uses 2/3 * 4 = 8/3 times the input dim for hidden layer
        #hidden_dim = int(8/3 * config.n_embd)
        hidden_dim = 2 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim)
        self.w2 = nn.Linear(config.n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim  , config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        with torch.cuda.amp.autocast(enabled=False):
            hidden = F.silu(x1) * x2  # SwiGLU activation
        x = self.c_proj(hidden)
        return x
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x1=self.ln_1(x)
        x = x + self.attn(x1)
        with torch.cuda.amp.autocast(enabled=False):
            x2=self.ln_2(x)
        x = x + self.mlp(x2)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # input embedding stem
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        with torch.cuda.amp.autocast(enabled=False):
            x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        logger.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 详细的参数统计
        logger.info("\n=== Parameter Analysis ===")
        
        # 1. 收集所有参数
        all_params = {pn: p for pn, p in self.named_parameters()}
        grad_params = {pn: p for pn, p in all_params.items() if p.requires_grad}
        
        logger.info(f"Total parameters: {len(all_params)}")
        logger.info(f"Parameters requiring gradient: {len(grad_params)}")
        if 0:
            # 2. 按模块分类统计
            module_params = {}
            for name, param in grad_params.items():
                module_name = name.split('.')[0]
                if module_name not in module_params:
                    module_params[module_name] = []
                module_params[module_name].append((name, param))
            
            for module_name, params in module_params.items():
                logger.info(f"\nModule: {module_name}")
                for param_name, param in params:
                    logger.info(f"  - {param_name}")
                    logger.info(f"    Shape: {param.shape}")
                    logger.info(f"    Dim: {param.dim()}")
            
        # 3. 分类参数
        nodecay_params=[]
        decay_params=[]
        
        for name, param in grad_params.items():
            # 检查是否是 `FlatParameter`
            if isinstance(param, FlatParameter):
                # 如果是 flat_param，根据原始参数名称来分组
                logger.info("\n=== flat_param in FSDP===")
                for orig_param in param._param_infos: 
                    orig_name = orig_param.module_name  
                    if "weight" in orig_name and not any(x in orig_name for x in ["ln_","bias", "LayerNorm", "layer_norm"]):
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)
            else:
                #logger.info(name)
                # 如果不是 flat_param，直接按常规方式分组
                if "weight" in name and not any(x in name for x in ["ln_","bias", "LayerNorm", "layer_norm"]):
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

        logger.info("\n=== Parameter Groups ===")
        logger.info(f"Decay parameters: {len(decay_params)}")
        logger.info(f"No-decay parameters: {len(nodecay_params)}")
        
        # 原有的优化器配置代码
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, fused=use_fused)
        
        return optimizer
    
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.split=split
        self.rng=np.random.default_rng(1337)
        # get the shard filenames
        data_root = "edu_fineweb10B_gpt4_enc"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            logger.info(f"found {len(shards)} shards for split {split}")
        self.reset()
    def load_tokens(self,filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32) # added after video
        shard = torch.tensor(npt, dtype=torch.long)
        if self.split == "train":
            # split tokens into documents using the <|endoftext|> token and shuffle
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents) # concatenate the documents back together
        return shard
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        # set the data loader to a specific state (from a checkpoint)
    def set_from_checkpoint(self, loader_checkpoint):
        self.current_position = loader_checkpoint['current_position'] + self.B * self.T * self.process_rank # we add the B*T*process_rank to the position to make sure it is the correct position for each process
        self.current_shard = loader_checkpoint['current_shard']
        self.tokens = self.load_tokens(self.shards[self.current_shard]) 
        # if loading the next batch will exceed the tokens, reset the position and load the next shard
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
    def next_batch(self):
        B, T = self.B, self.T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if master_process:
                logger.info(f"ddp_local_rank {ddp_local_rank} load new batch")
            if self.current_shard  == len(self.shards):
                self.rng.shuffle(self.shards)
            self.current_shard = (self.current_shard + 1) % len(self.shards)

            del self.tokens
            gc.collect()  # 强制进行垃圾回收

            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        #buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # buff to device then x,y is two views of buf, to avoid memory duplicate.
        buf = self.tokens[self.current_position : self.current_position + B * T + 1].to(device)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        #data_size = x.element_size() * x.nelement() 
        #if master_process:
        #    logger.info(f"ddp_local_rank Loading new data size: {data_size / 1024**2:.2f} MB")
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=2 train_gpt2.py

# run the training loop

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
def setup_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1  # 是否为分布式运行
    if ddp:
        # FSDP 需要 CUDA
        assert torch.cuda.is_available(), "FSDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        # 非分布式运行
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, device_type

ddp, ddp_rank, ddp_local_rank, ddp_world_size,  device, device_type = setup_distributed()
logger.info(f"ddp={ddp}, ddp_rank={ddp_rank}, ddp_local_rank={ddp_local_rank}, ddp_world_size={ddp_world_size}, device={device}, device_type={device_type}")
dtype = get_supported_dtype(device)

def get_fsdp_config(model_config):
    """Returns FSDP configuration"""
    
    # 1. 定义要包装的模块类型
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block,  # 确保这是你的 Block 类
            #CausalSelfAttention,  # 添加注意力层
            #MLP,  # 添加 MLP 层
        }
    )

    # 2. 配置混合精度
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        # 保持这些在 float32 以提高稳定性
        reduce_dtype=torch.float32,  
        buffer_dtype=torch.float32,
        keep_low_precision_grads=True
    )

    # 3. FSDP 配置
    fsdp_config = {
        "mixed_precision": mixed_precision_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # 或者考虑 SHARD_GRAD_OP
        #"auto_wrap_policy": transformer_wrap_policy,
        #"device_id": torch.cuda.current_device(),
        #"sync_module_states": True,  # 确保模块状态同步
        #"forward_prefetch": True,
        #"backward_prefetch": BackwardPrefetch.BACKWARD_POST,
        #"limit_all_gathers": True,
    }

    return fsdp_config


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#enc = tiktoken.get_encoding("gpt2")
#instead of use gpt2 tokenizer and vocabulary, we make a hack--use cl100k_base tokenizer and vocabulary
enc = tiktoken.get_encoding("cl100k_base")
max_lr = 5e-4
min_lr = max_lr * 0.1
warmup_steps = 2000 # 715 #+2000
max_steps = 2*19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

model_config=GPTConfig(n_layer=12, n_head=16, n_embd=1024,vocab_size=100288)

# create the log directory we will write checkpoints to and log to
fsdp_config=get_fsdp_config(model_config)
if ddp:
    logger.info(f"init FSDP:{fsdp_config=}")

#model_config=GPTConfig(vocab_size=50304)
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B =8 # micro batch size
T = model_config.block_size # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    logger.info(f"total desired batch size: {total_batch_size}")
    logger.info(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
current_step = 0
val_loss_accum=0.0
torch.set_float32_matmul_precision('high')

def load_model_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=device)
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint

# Resume or start new training
resume_training = False
if resume_training:
    checkpoint_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    assert len(checkpoint_files) > 0, "no checkpoints found to resume training from"
    checkpoint_file = sorted(checkpoint_files)[-1]
    # Create model and optimizer
    model = GPT(model_config)
    model.to(device)
    if ddp:
        model = FSDP(model, **fsdp_config,use_orig_params=True,auto_wrap_policy=size_based_auto_wrap_policy)
    
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    optimizer = model.module.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)
    
    # Load checkpoint
    checkpoint = load_model_checkpoint(model, optimizer, os.path.join(log_dir, checkpoint_file))
    current_step = checkpoint['step'] + 1
    train_loader.set_from_checkpoint(checkpoint['train_loader'])
    if master_process:
        logger.info(f"resuming training from {checkpoint_file=} with  {current_step=} with a validation loss of {checkpoint['val_loss']:.4f}")
else:
    # Create new model
    model = GPT(model_config)
    model.to(device)
    if ddp:
        model = FSDP(model, use_orig_params=True) #,  , , ,**fsdp_config
        #model = DDP(model,device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    logger.info(f"training from scratch with {model_config=}")

    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)
    
use_compile = False #True   # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)

logger.info(f"{use_compile=}")

scaler = torch.cuda.amp.GradScaler()

def train():
    global scaler 
    for step in range(current_step,max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        #print(f"step {step}")
        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if master_process:
                logger.info(f"#{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):

                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state = model.state_dict()
                    optimizer_state = FSDP.optim_state_dict(model, optimizer)
                    checkpoint = {
                        'model': model_state,
                        'optimizer': optimizer_state,
                        'config': model.module.config if hasattr(model, 'module') else model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'train_loader': {
                            'current_shard': train_loader.current_shard,
                            'current_position': train_loader.current_position
                        },
                    }
                    torch.save(checkpoint, checkpoint_path)

            #logger.info(f"done with val")
        # once in a while evaluate hellaswag
        if (step % 250 == 0 or last_step) and (not use_compile):
            #logger.info(f"in eval hellaswag 1")
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                #logger.info(f"in eval hellaswag 1 {i=}")
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                #logger.info(f"in eval hellaswag 2")
                _, tokens, mask, label = render_example(example,enc)
                #logger.info(f"in eval hellaswag 3")
                tokens = tokens.to(device)
                mask = mask.to(device)
                #logger.info(f"in eval hellaswag 4")
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                #logger.info(f"in eval hellaswag 5")
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            #logger.info(f"in eval hellaswag 6")
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                #dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                #dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
                #logger.info(f"in eval hellaswag 7")
            acc_norm = num_correct_norm / num_total
            if master_process:
                #logger.info(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                logger.info(f"#{step} hella {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # logger.info the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                logger.info(f"rank {ddp_rank} sample {i}: {decoded}")

        # do one step of the optimization
        
        model.train()
        torch.cuda.empty_cache()
        # 在梯度累积开始前清零梯度
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        
        # 使用 no_sync 上下文管理器进行梯度累积
        for micro_step in range(grad_accum_steps):
            grad_context = ( nullcontext() if  micro_step == grad_accum_steps - 1 else model.no_sync() )
            x, y = train_loader.next_batch()
            # 使用自动混合精度
            with grad_context:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, loss = model(x, y)
                # 缩放损失以考虑梯度累积
                loss = loss / grad_accum_steps
                
                # 记录损失值
                loss_accum += loss.detach()
                
                # 反向传播
                scaler.scale(loss).backward()

        if ddp and isinstance(model, FSDP):
            torch.cuda.synchronize()
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # 梯度裁剪前进行 unscale
        scaler.unscale_(optimizer)
        
        # 梯度裁剪
        norm=model.clip_grad_norm_(1.0)
        #norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新学习率
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 优化器步骤
        scaler.step(optimizer)
        # reset scaler
        if norm < 0.2 and norm*lr < 0.00001:  # 可根据需求调整重置条件
            scaler.update(new_scale=scaler.get_scale() * 2)
        else:
            scaler.update()
        
        # 确保 CUDA 操作完成
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            str=f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            logger.debug(f"{step} train {loss_accum.item():.6f}" )
            logger.info(str  )
            
try:
    train()
finally:
    #=== clear up ===
    if ddp:
        destroy_process_group()
    logging.shutdown()