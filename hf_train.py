from transformers import (
    Trainer, 
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import os

def setup_optimized_training():
    # 1. 定义训练参数，包含各种优化设置
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        
        # 分布式训练设置
        ddp_find_unused_parameters=False,  # 提高DDP效率
        ddp_bucket_cap_mb=25,  # DDP梯度同步bucket大小
        
        # 数据加载优化
        dataloader_num_workers=4,  # 多进程数据加载
        dataloader_pin_memory=True,  # 使用固定内存加速
        dataloader_prefetch_factor=2,  # 预加载batch数量
        
        # 内存优化
        gradient_accumulation_steps=4,  # 梯度累积
        max_grad_norm=1.0,  # 梯度裁剪
        gradient_checkpointing=True,  # 激活梯度检查点以节省显存
        
        # 混合精度训练
        fp16=True,  # 或者使用 bf16=True 用于更新的GPU
        fp16_opt_level="O2",  # apex优化级别
        
        # 缓存优化
        local_rank=-1,  # 自动检测本地rank
        dataloader_drop_last=True,  # 丢弃不完整的最后一个batch
        
        # 性能监控
        logging_dir="./logs",
        logging_steps=100,
        
        # 优化器设置
        learning_rate=5e-5,
        warmup_steps=500,
        
        # 评估和保存策略
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        
        # 硬件优化
        torch_compile=True,  # 使用PyTorch 2.0编译功能
        use_cpu=False  # 强制使用GPU
    )
    
    # 2. 创建模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 3. 准备数据集和数据整理器
    def prepare_dataset():
        # 示例数据集准备
        texts = ["Your text data here..."]
        dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,  # 并行处理
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    # 4. 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 5. 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepare_dataset(),
        data_collator=data_collator,
    )
    
    # 6. 自定义数据加载器优化（如果需要）
    def get_optimized_dataloader(dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            prefetch_factor=training_args.dataloader_prefetch_factor,
            persistent_workers=True,  # 保持worker进程存活
            drop_last=True,
        )
    
    # 7. 应用自定义数据加载器（可选）
    trainer.get_train_dataloader = lambda: get_optimized_dataloader(
        trainer.train_dataset, 
        training_args.per_device_train_batch_size
    )
    
    return trainer

# 使用示例
if __name__ == "__main__":
    # 设置环境变量以启用性能优化
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 启用异步CUDA操作
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 启用tokenizer并行
    
    # 创建并启动优化后的训练器
    trainer = setup_optimized_training()
    
    # 开始训练
    trainer.train()
    