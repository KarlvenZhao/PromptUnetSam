# nnUNet 服务器部署和训练流程指南

## 概述
本文档提供了在服务器上部署和运行nnUNet进行医学图像分割的完整流程。

## 环境要求

### 硬件要求
- GPU: NVIDIA GPU (推荐RTX 3080/4080或更高)
- 内存: 至少32GB RAM (推荐64GB+)
- 存储: 至少100GB可用空间
- CUDA: 支持CUDA 11.0+

### 软件依赖
```bash
# Python环境
Python >= 3.8
conda或miniconda

# 核心依赖包
torch >= 1.12.0
torchvision
numpy
scipy
scikit-image
scikit-learn
batchgenerators
nnunetv2
```

## 安装步骤

### 1. 创建conda环境
```bash
conda create -n nnunet python=3.9
conda activate nnunet
```

### 2. 安装PyTorch
```bash
# 根据CUDA版本选择合适的PyTorch版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. 安装nnUNet
```bash
pip install nnunetv2
```

### 4. 设置环境变量
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## 数据准备流程

### 1. 数据格式要求
```
nnUNet_raw/
└── Dataset999_MedSAM_MR_Brain/
    ├── dataset.json
    ├── imagesTr/
    │   ├── MR_Brain_001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── MR_Brain_001.nii.gz
    │   └── ...
    └── imagesTs/
        ├── MR_Brain_test_001_0000.nii.gz
        └── ...
```

### 2. 数据转换脚本
使用项目中的转换脚本:
```bash
python tmp/run_preprocess_and_analyze.py
```

### 3. 验证数据格式
```bash
nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity
```

## 训练流程

### 1. 数据预处理
```bash
# 自动规划和预处理数据
nnUNetv2_plan_and_preprocess -d 999
```

### 2. 开始训练
```bash
# 2D训练 (推荐用于MRI脑部数据)
nnUNetv2_train 999 2d 0

# 如果内存不足，使用npz格式
nnUNetv2_train 999 2d 0 --npz

# 3D训练 (如果数据支持)
nnUNetv2_train 999 3d_fullres 0
```

### 3. 交叉验证训练
```bash
# 训练所有5个fold
for fold in {0..4}; do
    nnUNetv2_train 999 2d $fold
done
```

### 4. 模型集成
```bash
# 集成多个fold的结果
nnUNetv2_ensemble -i fold_0 fold_1 fold_2 fold_3 fold_4 -o ensemble_output
```

## 推理流程

### 1. 单模型推理
```bash
nnUNetv2_predict -i input_folder -o output_folder -d 999 -c 2d -f 0
```

### 2. 集成模型推理
```bash
nnUNetv2_predict -i input_folder -o output_folder -d 999 -c 2d -f 0 1 2 3 4
```

## 监控和调试

### 1. 训练监控
```bash
# 查看训练日志
tail -f nnUNet_results/Dataset999_MedSAM_MR_Brain/nnUNetTrainer__nnUNetPlans__2d/fold_0/training_log_*.txt

# 使用tensorboard (如果可用)
tensorboard --logdir nnUNet_results/Dataset999_MedSAM_MR_Brain/
```

### 2. 常见问题解决

#### 内存不足
```bash
# 减少批次大小
export nnUNet_batch_size=2

# 使用npz格式
nnUNetv2_train 999 2d 0 --npz

# 减少工作进程数
export nnUNet_n_proc_DA=4
```

#### CUDA内存不足
```bash
# 设置较小的patch size
export nnUNet_def_n_proc=2

# 使用混合精度训练
nnUNetv2_train 999 2d 0 --use_compressed_data
```

## 性能优化

### 1. 数据加载优化
```bash
# 使用SSD存储预处理数据
# 增加数据加载工作进程数
export nnUNet_n_proc_DA=8
```

### 2. GPU优化
```bash
# 启用CUDA优化
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
```

## 必需文件清单

### 核心文件
- `tmp/run_preprocess_and_analyze.py` - 数据预处理和转换脚本
- `tmp/convert_medsam_to_nnunet.py` - MedSAM到nnUNet格式转换
- `data/sam_mri_processed_relaxed/` - 原始MRI数据目录

### 配置文件
- `dataset.json` - 数据集配置文件
- `nnUNetPlans.json` - 自动生成的训练计划文件

### 环境配置
```bash
# requirements.txt
torch>=1.12.0
torchvision
numpy
scipy
scikit-image
scikit-learn
batchgenerators
nnunetv2
nibabel
SimpleITK
```

## 部署脚本示例

### 完整部署脚本 (deploy_nnunet.sh)
```bash
#!/bin/bash

# 设置环境变量
export nnUNet_raw="/data/nnUNet_raw"
export nnUNet_preprocessed="/data/nnUNet_preprocessed"
export nnUNet_results="/data/nnUNet_results"

# 激活环境
conda activate nnunet

# 数据预处理
echo "开始数据预处理..."
python tmp/run_preprocess_and_analyze.py

# 验证数据
echo "验证数据格式..."
nnUNetv2_plan_and_preprocess -d 999 --verify_dataset_integrity

# 开始训练
echo "开始训练..."
nnUNetv2_train 999 2d 0 --npz

echo "训练完成！"
```

## 注意事项

1. **内存管理**: MRI数据通常较大，确保有足够的RAM和GPU内存
2. **存储空间**: 预处理数据会占用大量空间，预留足够存储
3. **训练时间**: 完整训练可能需要数小时到数天，建议使用screen或tmux
4. **数据备份**: 定期备份训练检查点和重要数据
5. **版本兼容**: 确保PyTorch和CUDA版本兼容

## 故障排除

### 常见错误及解决方案

1. **RuntimeError: CUDA out of memory**
   - 减少批次大小
   - 使用--npz参数
   - 降低图像分辨率

2. **FileNotFoundError: dataset.json**
   - 检查数据路径设置
   - 确保转换脚本正确执行

3. **训练进程意外终止**
   - 检查系统内存使用
   - 查看训练日志文件
   - 减少并行工作进程数

## 联系和支持

如遇到问题，请检查:
1. nnUNet官方文档: https://github.com/MIC-DKFZ/nnUNet
2. 项目日志文件
3. 系统资源使用情况