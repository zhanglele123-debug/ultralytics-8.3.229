# ===================== 导入依赖库 =====================
import os
import sys
import yaml
import numpy as np
from ultralytics import YOLO
import torch
import traceback

# ===================== 核心：EW-MPDIoU 算法实现 =====================

def ew_mpdiou(pred_boxes, target_boxes, xywh=True, eps=1e-7):
    """
    Edge-Weighted Minimum Point Distance IoU (EW-MPDIoU)

    公式: EW-MPDIoU = IoU - w1 * d1^2 / c^2 - w2 * d2^2 / c^2

    d1 = 预测框与GT框左上角之间的欧氏距离
    d2 = 预测框与GT框右下角之间的欧氏距离
    c  = 最小包围框对角线长度
    w1, w2 = 基于GT框宽高比的边缘感知权重 (w1 + w2 = 2)

    对于宽框(w > h): w2 > 1 → 右下角惩罚更大 → x方向定位更精确
    对于高框(h > w): w1 > 1 → 左上角惩罚更大 → y方向定位更精确

    Args:
        pred_boxes:  (N, 4) 预测框
        target_boxes: (N, 4) GT框
        xywh: True=(x,y,w,h), False=(x1,y1,x2,y2)
        eps: 防止除零

    Returns:
        (N, 1) EW-MPDIoU值 (越接近1越好)
    """
    # 转换为 xyxy 格式
    if xywh:
        (x1, y1, w1, h1) = pred_boxes.chunk(4, -1)
        (x2, y2, w2, h2) = target_boxes.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 交集面积
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # 并集面积
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - inter + eps

    # IoU
    iou = inter / union

    # 最小包围框对角线平方
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # 角点距离平方
    d1_sq = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)  # 左上角
    d2_sq = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)  # 右下角

    # ========== 边缘感知权重 (Edge-Weighted 核心) ==========
    gt_w = b2_x2 - b2_x1  # GT框宽度
    gt_h = b2_y2 - b2_y1  # GT框高度
    w_ratio = gt_w / (gt_w + gt_h + eps)  # 宽占比: 宽框→大
    h_ratio = gt_h / (gt_w + gt_h + eps)  # 高占比: 高框→大
    # 宽框(w>h): w1小 w2大 → 右下角惩罚更重 → x方向更敏感
    # 高框(h>w): w1大 w2小 → 左上角惩罚更重 → y方向更敏感
    w1 = 2.0 * h_ratio  # 左上角权重
    w2 = 2.0 * w_ratio  # 右下角权重 (w1 + w2 ≡ 2.0)

    # EW-MPDIoU
    mpdiou = iou - (w1 * d1_sq + w2 * d2_sq) / c2
    return mpdiou


# ===================== 运行时替换：注入 EW-MPDIoU 到 BboxLoss =====================

_original_forward = None  # 备份，用于恢复


def patch_bbox_loss():
    """
    运行时替换 BboxLoss.forward()，将 CIoU 替换为 EW-MPDIoU。
    在 model.train() 之前调用即可，不修改任何 ultralytics 源码。
    """
    global _original_forward
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.tal import bbox2dist

    if _original_forward is not None:
        print("[EW-MPDIoU] 已经注入，跳过重复 patch")
        return

    _original_forward = BboxLoss.forward

    def ew_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                   target_scores, target_scores_sum, fg_mask):
        """使用 EW-MPDIoU 的 BboxLoss forward"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = ew_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL 损失保持不变
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    BboxLoss.forward = ew_forward
    print("[EW-MPDIoU] BboxLoss.forward 已替换为 EW-MPDIoU")


def restore_bbox_loss():
    """恢复原始的 CIoU BboxLoss"""
    global _original_forward
    if _original_forward is None:
        print("[EW-MPDIoU] 没有 patch 记录，无需恢复")
        return
    from ultralytics.utils.loss import BboxLoss
    BboxLoss.forward = _original_forward
    _original_forward = None
    print("[EW-MPDIoU] BboxLoss.forward 已恢复为原始 CIoU")


# ===================== 数据集配置 =====================

def create_data_yaml(data_dir="datasets/icon"):
    """自动扫描标签文件，生成YOLO数据集配置文件"""
    train_labels_dir = os.path.join(data_dir, "labels/train")
    val_labels_dir = os.path.join(data_dir, "labels/val")

    all_class_ids = set()

    for labels_dir in [train_labels_dir, val_labels_dir]:
        if os.path.exists(labels_dir):
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(labels_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                all_class_ids.add(int(parts[0]))

    print(f"检测到的类别ID: {sorted(list(all_class_ids))}")
    class_names = {class_id: f'class_{class_id}' for class_id in sorted(all_class_ids)}

    data_config = {
        'path': data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names,
        'nc': len(class_names)
    }

    yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"配置文件已生成: {yaml_path}")
    return yaml_path


# ===================== 核心训练 =====================

def train_yolo11n_ew():
    """
    基于 test.py 最优配置 + EW-MPDIoU 损失函数。
    适配 640x320 非正方形图标、小样本、Windows 环境。
    """
    data_dir = "datasets/icon"

    # 自动创建目录结构
    required_dirs = [
        os.path.join(data_dir, f"{a}/{b}")
        for a in ["images", "labels"] for b in ["train", "val"]
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 统计图片
    train_images = os.listdir(os.path.join(data_dir, "images/train"))
    val_images = os.listdir(os.path.join(data_dir, "images/val"))
    print(f"训练图: {len(train_images)} | 验证图: {len(val_images)}")

    # 生成数据集配置
    data_yaml = create_data_yaml(data_dir)

    # ========== 注入 EW-MPDIoU ==========
    patch_bbox_loss()

    # 加载模型
    print("加载YOLOv11n模型 (EW-MPDIoU)...")
    model = YOLO('yolo11n.pt')

    # 训练参数 (基于 test.py 最优配置)
    train_args = {
        'data': data_yaml,
        'epochs': 300,
        'batch': 32,
        'imgsz': 640,
        'rect': True,
        'workers': 0,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'pretrained': True,
        'optimizer': 'AdamW',
        'freeze': 10,
        'cache': True,
        'patience': 100,
        'amp': True,
        'verbose': True,
        'save': True,
        'save_period': 10,
        'name': 'yolo11n_ew_mpdiou',
        'exist_ok': True,
        'box': 10.0,
        'cls': 1.0,
        'dfl': 1.5,
        'lr0': 0.001,
        'lrf': 0.01,
        'warmup_epochs': 8.0,
        'weight_decay': 0.0005,
        'close_mosaic': 20,
        'resume': False,
        'mosaic': 0.3,
        'degrees': 0,
        'scale': 0.1,
        'translate': 0.05,
        'hsv_h': 0.005,
        'hsv_s': 0.2,
        'hsv_v': 0.2,
        'fliplr': 0.0,
    }

    print(f"\n{'='*50}")
    print(f"EW-MPDIoU 训练配置")
    print(f"损失函数: EW-MPDIoU (Edge-Weighted MPDIoU)")
    print(f"{'='*50}")

    try:
        results = model.train(**train_args)

        # 验证最优模型
        best_model = YOLO(f"runs/detect/{train_args['name']}/weights/best.pt")
        metrics = best_model.val(data=data_yaml)
        print(f"\n训练完成！mAP50: {metrics.box.map50:.4f}")

        # 导出 ONNX
        best_model.export(format='onnx', imgsz=640, simplify=True)
        print("ONNX 模型已导出")

        return results

    except Exception as e:
        print(f"错误：{e}")
        traceback.print_exc()

    finally:
        # 恢复原始 BboxLoss，避免影响后续使用
        restore_bbox_loss()


# ===================== 环境检查 =====================

def check_environment():
    print("检查环境...")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    try:
        import ultralytics
        YOLO('yolo11n.pt')
        print("环境正常！")
        return True
    except Exception:
        print("请安装：pip install ultralytics")
        return False


# ===================== 标签检查修复 =====================

def check_dataset_labels(data_dir="datasets/icon"):
    print("\n检查标签格式...")
    for split in ["train", "val"]:
        label_dir = os.path.join(data_dir, f"labels/{split}")
        if os.path.exists(label_dir):
            for f in os.listdir(label_dir):
                if f.endswith('.txt'):
                    path = os.path.join(label_dir, f)
                    with open(path, 'r+') as file:
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
    train_yolo11n_ew()
