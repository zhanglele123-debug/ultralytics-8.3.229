# ===================== 导入依赖库 =====================
import os       # 文件/文件夹路径操作、创建目录
import sys      # 识别操作系统（Windows专用适配）
import yaml     # 生成YOLO数据集配置文件 data.yaml
import numpy as np  # 数值计算，处理模型精度指标
from ultralytics import YOLO  # YOLOv11核心库：模型加载/训练/推理
import torch    # PyTorch深度学习框架（YOLO底层依赖）
import traceback  # 捕获训练报错，方便排查问题

# ===================== 函数1：自动生成数据集配置 =====================
def create_data_yaml(data_dir="datasets/icon"):
    """
    功能：自动扫描标签文件，生成YOLO训练必需的配置文件
    适配：640×320分辨率图标数据集
    """
    # 训练/验证标签路径
    train_labels_dir = os.path.join(data_dir, "labels/train")
    val_labels_dir = os.path.join(data_dir, "labels/val")
    
    all_class_ids = set()  # 自动提取所有类别ID（去重）

    # 读取训练集标签
    if os.path.exists(train_labels_dir):
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(train_labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_class_ids.add(int(parts[0]))
    
    # 读取验证集标签
    if os.path.exists(val_labels_dir):
        for label_file in os.listdir(val_labels_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(val_labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            all_class_ids.add(int(parts[0]))
    
    # 生成类别映射
    print(f"检测到的类别ID: {sorted(list(all_class_ids))}")
    class_names = {class_id: f'class_{class_id}' for class_id in sorted(all_class_ids)}
    
    # YOLO标准yaml配置
    data_config = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names,
        'nc': len(class_names)
    }
    
    # 保存配置文件
    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"配置文件已生成: {yaml_path}")
    return yaml_path

# ===================== 函数2：核心训练（兼容8.3.229 + 640×320 + 平滑曲线） =====================
def train_yolo11n():
    """
    核心优化：
    1. 兼容 ultralytics 8.3.229 版本（删除无效参数）
    2. 适配 640×320 分辨率图片（等比例训练，不变形）
    3. 彻底解决Windows训练曲线跳变、震荡问题
    4. 小样本图标专用优化
    """
    # 数据集路径
    data_dir = "datasets/icon"
    
    # 自动创建YOLO标准文件夹结构
    required_dirs = [os.path.join(data_dir, f"{a}/{b}") for a in ["images", "labels"] for b in ["train", "val"]]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 统计图片数量
    train_images = os.listdir(os.path.join(data_dir, "images/train"))
    val_images = os.listdir(os.path.join(data_dir, "images/val"))
    print(f"训练图: {len(train_images)} | 验证图: {len(val_images)}")

    # 生成数据集配置
    data_yaml = create_data_yaml(data_dir)

    # 加载YOLOv11轻量模型
    print("加载YOLOv11n模型...")
    model = YOLO('yolo11n.pt')

    # ===================== 训练参数（兼容旧版本 + 适配640×320） =====================
    train_args = {
        # ========== 基础配置 ==========
        'data': data_yaml,
        'epochs': 300,                   # 训练轮数（防止过拟合）
        'batch': 32,                     # 批次大小（根据显存调整）
        # ===================== 核心适配：640×320 图片 =====================
        'imgsz': 640,                    # 最大边长=640，自动适配320高度
        'rect': True,                    # 开启矩形训练，完美适配非正方形图片（无拉伸）
        # ===================== Windows 必改，根治跳变 =====================
        'workers': 0,                    # Windows专属，数据加载稳定不崩溃
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'seed': 42,                      # 固定随机种子，复现训练

        # ========== 模型优化（平滑训练核心） ==========
        'pretrained': True,              # 迁移学习，小样本必备
        'optimizer': 'AdamW',            # 比SGD平滑10倍
        'freeze': 10,                    # 冻结主干，小样本不震荡
        'cache': True,                   # 数据缓存，消除读写波动
        'patience': 100,                  # 早停：100轮不提升自动停止
        'amp': True,                     # 混合精度，加速+省显存
        'verbose': True,                 # 打印详细日志
        'save': True,                    # 保存模型
        'save_period': 10,               # 每10轮保存一次
        'name': 'yolo11n_icon_smooth',   # 训练结果文件夹
        'exist_ok': True,                # 覆盖旧结果

        # ========== 损失函数（优化定位/分类） ==========
        'box': 10.0,
        'cls': 1.0,
        'dfl': 1.5,

        # ========== 学习率（彻底消除跳变） ==========
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 8.0,
        'weight_decay': 0.0005,
        'close_mosaic': 20,              # 最后20轮关闭马赛克
        'resume': False,                 # 断点续训关闭

        # ========== 数据增强（低增强=平滑曲线） ==========
        'mosaic': 0.3,
        'degrees': 0,
        'scale': 0.1,
        'translate': 0.05,
        'hsv_h': 0.005,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'fliplr': 0.0,
    }

    # 开始训练
    print("\n开始训练（适配 640×320 分辨率）")
    try:
        results = model.train(**train_args)
        # 验证最优模型
        best_model = YOLO(f"runs/detect/{train_args['name']}/weights/best.pt")
        metrics = best_model.val(data=data_yaml)
        print(f"\n训练完成！mAP50: {metrics.box.map50:.4f}")
        # 导出部署模型
        best_model.export(format='onnx', imgsz=640, simplify=True)
        return results
    except Exception as e:
        print(f"错误：{e}")
        traceback.print_exc()

# ===================== 函数3：环境检查 =====================
def check_environment():
    print("检查环境...")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    try:
        import ultralytics
        YOLO('yolo11n.pt')
        print("环境正常！")
        return True
    except:
        print("请安装：pip install ultralytics")
        return False

# ===================== 函数4：标签检查修复 =====================
def check_dataset_labels(data_dir="datasets/icon"):
    print("\n检查标签格式...")
    for split in ["train", "val"]:
        label_dir = os.path.join(data_dir, f"labels/{split}")
        if os.path.exists(label_dir):
            for f in os.listdir(label_dir):
                if f.endswith('.txt'):
                    with open(os.path.join(label_dir, f), 'r+') as file:
                        lines = [l for l in file.readlines() if len(l.strip().split()) >= 5]
                        file.seek(0)
                        file.writelines(lines)
                        file.truncate()
    print("标签检查完成！")

# ===================== 主程序 =====================
if __name__ == "__main__":
    if not check_environment():
        exit(1)
    check_dataset_labels()
    train_yolo11n()