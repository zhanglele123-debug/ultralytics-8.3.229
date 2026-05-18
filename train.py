import os
import yaml
import numpy as np
from ultralytics import YOLO
import torch

def create_data_yaml(data_dir="datasets/icon"):
    """创建数据集配置文件"""
    
    # 首先检查标签文件，确定实际的类别数量
    train_labels_dir = os.path.join(data_dir, "labels/train")
    val_labels_dir = os.path.join(data_dir, "labels/val")
    
    all_class_ids = set()
    
    # 检查训练标签中的类别
    if os.path.exists(train_labels_dir):
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_class_ids.add(int(parts[0]))
    
    # 检查验证标签中的类别
    if os.path.exists(val_labels_dir):
        for label_file in os.listdir(val_labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(val_labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_class_ids.add(int(parts[0]))
    
    print(f"检测到的类别ID: {sorted(list(all_class_ids))}")
    
    # 创建类别名称映射
    class_names = {}
    for class_id in sorted(all_class_ids):
        class_names[class_id] = f'class_{class_id}'
    
    data_config = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names,
        'nc': len(class_names)  # 类别数量
    }
    
    # 保存为yaml文件
    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"数据配置文件创建于: {yaml_path}")
    print(f"类别配置: {class_names}")
    
    return yaml_path

def train_yolo11n():
    """训练YOLOv11n模型"""
    # 设置数据集路径
    data_dir = "datasets/icon"
    
    # 检查数据集结构
    required_dirs = [
        os.path.join(data_dir, "images/train"),
        os.path.join(data_dir, "images/val"),
        os.path.join(data_dir, "labels/train"),
        os.path.join(data_dir, "labels/val")
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"警告: 目录不存在 {dir_path}")
            # 创建缺失的目录
            os.makedirs(dir_path, exist_ok=True)
    
    # 检查是否有训练和验证图像
    train_images = os.listdir(os.path.join(data_dir, "images/train"))
    val_images = os.listdir(os.path.join(data_dir, "images/val"))
    
    if not train_images:
        print("警告: 训练集图像为空!")
    if not val_images:
        print("警告: 验证集图像为空!")
    
    print(f"训练图像数量: {len(train_images)}")
    print(f"验证图像数量: {len(val_images)}")
    
    # 创建数据配置文件
    data_yaml = create_data_yaml(data_dir)
    
    # 加载YOLOv11n模型
    print("加载YOLOv11n模型...")
    model = YOLO('yolo11n.pt')  # 使用官方预训练权重
    
    # 训练参数配置
    train_args = {
        'data': data_yaml,           # 数据配置文件路径
        'epochs': 400,               # 训练轮数
        'batch': 16,                 # 批次大小
        'imgsz': 640,                # 图像尺寸
        'workers': 8,                # 数据加载工作进程数
        'device': '0' if torch.cuda.is_available() else 'cpu',  # 使用GPU如果可用
        'seed': 42,                  # 随机种子
        'pretrained': True,          # 使用预训练权重
        'optimizer': 'auto',         # 自动选择优化器
        'verbose': True,             # 显示详细输出
        'save': True,                # 保存训练结果
        'save_period': 10,           # 每10轮保存一次检查点
        'cache': False,              # 缓存数据（如果内存足够可以设为True）
        'name': 'yolo11n_icon',      # 实验名称
        'exist_ok': True,            # 允许覆盖现有实验
        'patience': 50,              # 早停耐心值
        'box': 7.5,                  # 边界框损失权重
        'cls': 0.5,                  # 分类损失权重
        'dfl': 1.5,                  # DFL损失权重
        'close_mosaic': 10,          # 最后10轮关闭马赛克增强
        'resume': False,             # 是否从检查点恢复训练
        'amp': True,                 # 自动混合精度训练
        'fraction': 1.0,             # 使用全部数据
        'lr0': 0.002,                 # 初始学习率
        'lrf': 0.005,                 # 最终学习率因子
        'momentum': 0.937,           # 动量
        'weight_decay': 0.0005,      # 权重衰减
        'warmup_epochs': 12.0,        # 预热轮数
        'warmup_momentum': 0.8,      # 预热动量
        'warmup_bias_lr': 0.1,       # 预热偏置学习率
    }
    
    # 显示训练信息
    print("\n" + "="*50)
    print("训练配置:")
    print(f"数据集: {data_dir}")
    print(f"训练轮数: {train_args['epochs']}")
    print(f"批次大小: {train_args['batch']}")
    print(f"图像尺寸: {train_args['imgsz']}")
    print(f"设备: {train_args['device']}")
    print(f"输出目录: runs/detect/{train_args['name']}")
    print("="*50 + "\n")
    
    # 开始训练
    print("开始训练YOLOv11n...")
    try:
        results = model.train(**train_args)
        
        # 打印训练结果摘要
        print("\n" + "="*50)
        print("训练完成!")
        print("最佳模型保存在: runs/detect/{}/weights/best.pt".format(train_args['name']))
        
        # 验证最佳模型
        print("\n验证最佳模型...")
        best_model_path = f"runs/detect/{train_args['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            metrics = best_model.val(data=data_yaml, split='val')
            
            # 安全地打印指标
            print("\n验证指标:")
            print(f"mAP50-95: {metrics.box.map:.4f}" if hasattr(metrics.box, 'map') and metrics.box.map is not None else "mAP50-95: N/A")
            print(f"mAP50: {metrics.box.map50:.4f}" if hasattr(metrics.box, 'map50') and metrics.box.map50 is not None else "mAP50: N/A")
            
            # 处理可能为numpy数组的精确率和召回率
            if hasattr(metrics.box, 'p'):
                if isinstance(metrics.box.p, np.ndarray):
                    print(f"平均精确率: {metrics.box.p.mean():.4f}")
                    for i, p in enumerate(metrics.box.p):
                        print(f"  类别 {i} 精确率: {p:.4f}")
                else:
                    print(f"精确率: {metrics.box.p:.4f}")
            
            if hasattr(metrics.box, 'r'):
                if isinstance(metrics.box.r, np.ndarray):
                    print(f"平均召回率: {metrics.box.r.mean():.4f}")
                    for i, r in enumerate(metrics.box.r):
                        print(f"  类别 {i} 召回率: {r:.4f}")
                else:
                    print(f"召回率: {metrics.box.r:.4f}")
        
        # 导出模型（可选）
        print("\n导出模型...")
        try:
            export_path = best_model.export(format='onnx', imgsz=640, simplify=True)
            print(f"ONNX模型已导出到: {export_path}")
        except Exception as export_error:
            print(f"模型导出失败: {export_error}")
        
        return results
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_environment():
    """检查环境和依赖"""
    print("检查环境...")
    
    # 检查PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查ultralytics版本
    try:
        import ultralytics
        print(f"Ultralytics版本: {ultralytics.__version__}")
        
        # 检查YOLOv11可用性
        try:
            model = YOLO('yolo11n.pt')
            print("YOLOv11模型加载成功")
        except Exception as e:
            print(f"YOLOv11模型加载失败: {e}")
            print("尝试使用YOLOv8作为替代...")
            return 'yolov8n.pt'
        
        return 'yolo11n.pt'
    except ImportError as e:
        print("错误: 未安装ultralytics库")
        print("请运行: pip install ultralytics")
        return None

def check_dataset_labels(data_dir="datasets/icon"):
    """检查数据集标签文件，修复可能的类别ID问题"""
    print("\n检查数据集标签...")
    
    label_dirs = [
        os.path.join(data_dir, "labels/train"),
        os.path.join(data_dir, "labels/val")
    ]
    
    for label_dir in label_dirs:
        if os.path.exists(label_dir):
            fixed_count = 0
            for label_file in os.listdir(label_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(label_dir, label_file)
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # 检查类别ID是否为整数
                            try:
                                class_id = int(parts[0])
                                # 如果类别ID太大，可能需要调整
                                if class_id > 10:  # 假设最多10个类别
                                    print(f"警告: {label_file} 中的类别ID {class_id} 可能过大")
                                new_lines.append(line)
                            except ValueError:
                                print(f"错误: {label_file} 中的类别ID格式错误: {parts[0]}")
                    
                    # 如果文件有内容，重新写入
                    if new_lines:
                        with open(label_path, 'w') as f:
                            f.writelines(new_lines)
            
            print(f"检查完成: {label_dir}")

if __name__ == "__main__":
    # 检查环境
    model_type = check_environment()
    if not model_type:
        exit(1)
    
    # 检查数据集标签
    check_dataset_labels()
    
    # 开始训练
    train_yolo11n()