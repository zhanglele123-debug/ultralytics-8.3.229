# -*- coding: utf-8 -*-
"""
地铁巡检定位验证系统
适配你的YOLO模型类别：class_0=轨枕, class_1=喷号
"""

import os
import re
import sys
import csv
import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QLineEdit, QComboBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox,
    QHeaderView
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# sleeper面积阈值。0.50表示小于当前帧最大sleeper框面积50%的sleeper不参与计数。
SLEEPER_AREA_RATIO_THRES = 0.50


def current_timestamp_ms() -> str:
    """返回13位毫秒时间戳，便于表格记录和导出。"""
    return str(int(time.time() * 1000))


def natural_sort_key(path_or_name):
    """文件名自然排序，避免10.jpg排在2.jpg前面"""
    name = os.path.basename(str(path_or_name))
    stem, ext = os.path.splitext(name)

    parts = re.split(r"(\d+)", stem)
    key = []
    for part in parts:
        if part == "":
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))

    key.append((1, ext.lower()))
    return key


def normalize_spray_text(text: str) -> str:
    """清洗OCR字符串，保留字母数字"""
    if text is None:
        return ""

    text = str(text).upper()

    replace_map = {
        "O": "0", "I": "1", "L": "1", "Z": "2", "B": "8",
        " ": "", "\n": "", "\t": "", "-": "", "_": "", ".": "",
        ":": "", "：": "", "+": "", "$": "", "￥": "", "#": "",
        "*": "", "/": "", "\\": "", "|": "",
    }

    for k, v in replace_map.items():
        text = text.replace(k, v)

    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def parse_spray_number(text: str):
    """解析喷号，支持S/X/K前缀"""
    clean = normalize_spray_text(text)

    # 格式1：S/X + 可选K + 两位公里标 + 四位计数值
    match = re.search(r"([SX])K?(\d{2})(\d{4})", clean)
    if match:
        direction_prefix = match.group(1)
        km = int(match.group(2))
        count_value = int(match.group(3))
        direction = "上行" if direction_prefix == "S" else "下行"
        full_text = match.group(0)

        return {
            "raw": text, "clean": clean,
            "direction_prefix": direction_prefix, "direction": direction,
            "full_text": full_text, "km": km, "km_text": f"K{km:02d}",
            "count_value": count_value, "count_text": f"{count_value:04d}",
        }

    # 格式2：只有K前缀
    match = re.search(r"K(\d{2})(\d{4})", clean)
    if match:
        km = int(match.group(1))
        count_value = int(match.group(2))
        full_text = match.group(0)

        return {
            "raw": text, "clean": clean,
            "direction_prefix": "", "direction": None,
            "full_text": full_text, "km": km, "km_text": f"K{km:02d}",
            "count_value": count_value, "count_text": f"{count_value:04d}",
        }

    return None


def to_overlay_ascii(text: str) -> str:
    """转换为OpenCV支持的ASCII文本"""
    if text is None:
        return ""
    text = str(text).replace("上行", "UP").replace("下行", "DOWN")
    text = text.replace("站", "")
    text = re.sub(r"[^A-Za-z0-9_+\-:./ ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "SECTION"


def class_name_from_result(names, cls_id: int) -> str:
    """兼容不同版本Ultralytics的类别名格式"""
    try:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and cls_id < len(names):
            return str(names[cls_id])
    except Exception:
        pass
    return str(cls_id)


def safe_crop(img, box, pad=6):
    """安全裁剪喷号区域"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def bbox_area(box):
    """计算检测框面积"""
    x1, y1, x2, y2 = map(int, box)
    return max(0, x2 - x1) * max(0, y2 - y1)


def draw_box(img, box, label, color):
    """绘制检测框"""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img, str(label), (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
    )


class DetectionWorker(QThread):
    frame_signal = pyqtSignal(object)
    status_signal = pyqtSignal(dict)
    record_signal = pyqtSignal(list)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, source_path, conf_thres=0.25, iou_thres=0.45, tcms_info=None, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.source_path = source_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.running = True

        self.model = None
        self.ocr = None

        self.total_count_value = 0
        self.current_km = 0
        self.current_count_value = 0
        self.current_direction = "未识别"
        self.current_section = "未知区间"
        self.last_valid_spray = ""
        self.last_ocr_text = ""

        self.tcms_info = tcms_info or {}
        current_station = self.tcms_info.get("current_station", "A站")
        next_station = self.tcms_info.get("next_station", "B站")
        self.current_section = f"{current_station}-{next_station}"

        self.processed_image_paths = set()

    def stop(self):
        self.running = False

    def load_models(self):
        """加载YOLO和PP-OCR"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.log_signal.emit(f"YOLO模型加载完成：{self.model_path}")
        except Exception as e:
            raise RuntimeError(f"YOLO模型加载失败：{e}")

        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self.log_signal.emit("PP-OCR加载完成")
        except Exception as e:
            self.ocr = None
            self.log_signal.emit(f"PP-OCR加载失败：{e}")

    def list_sources(self):
        """读取图像文件夹并排序"""
        if not os.path.isdir(self.source_path):
            raise ValueError("当前版本只支持选择图像文件夹")

        files = []
        for name in os.listdir(self.source_path):
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(self.source_path, name))

        files = sorted(files, key=natural_sort_key)
        if len(files) == 0:
            raise ValueError("所选文件夹内没有可读取的图像文件")
        return files

    def run(self):
        try:
            self.load_models()
            image_files = self.list_sources()
            self.log_signal.emit(f"共读取到 {len(image_files)} 张图像")
            order_preview = "、".join([os.path.basename(p) for p in image_files[:10]])
            self.log_signal.emit(f"识别顺序前10张：{order_preview}")

            for img_path in image_files:
                if not self.running:
                    break

                abs_img_path = os.path.abspath(img_path)
                if abs_img_path in self.processed_image_paths:
                    self.log_signal.emit(f"跳过重复图像：{os.path.basename(img_path)}")
                    continue
                self.processed_image_paths.add(abs_img_path)

                frame = cv2.imread(img_path)
                if frame is None:
                    self.log_signal.emit(f"图像读取失败：{img_path}")
                    continue

                self.process_frame(frame, image_name=os.path.basename(img_path))
                time.sleep(0.03)

        except Exception as e:
            self.log_signal.emit("程序运行出错：\n" + str(e))
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.finished_signal.emit()

    def run_ocr(self, crop):
        """对喷号裁剪图运行OCR"""
        if crop is None or crop.size == 0 or self.ocr is None:
            return ""

        try:
            result = self.ocr.ocr(crop, cls=True)
            texts = []
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, list):
                        for line in item:
                            if isinstance(line, list) and len(line) >= 2:
                                rec = line[1]
                                if isinstance(rec, (tuple, list)) and len(rec) >= 1:
                                    texts.append(str(rec[0]))
            return "".join(texts)
        except Exception as e:
            self.log_signal.emit(f"OCR识别失败：{e}")
            return ""

    def process_frame(self, frame, image_name=""):
        """核心处理：适配你的模型类别（class_0=轨枕, class_1=喷号）"""
        show_img = frame.copy()

        result = self.model.predict(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)[0]
        names = result.names
        boxes = result.boxes

        raw_sleeper_box_count = 0
        valid_sleeper_box_count = 0
        frame_count_value = 0
        ocr_text_this_frame = ""
        parsed_spray = None
        detect_count = {}
        detections = []
        sleeper_areas = []

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = class_name_from_result(names, cls_id)
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                area = bbox_area(xyxy)

                detect_count[cls_name] = detect_count.get(cls_name, 0) + 1
                detections.append({"cls_name": cls_name, "conf": conf, "xyxy": xyxy, "area": area})

                # 适配你的模型：class_0 是轨枕
                if cls_name == "class_0":
                    sleeper_areas.append(area)

        raw_sleeper_box_count = len(sleeper_areas)
        max_sleeper_area = max(sleeper_areas) if sleeper_areas else 0
        sleeper_area_threshold = max_sleeper_area * SLEEPER_AREA_RATIO_THRES if max_sleeper_area > 0 else 0

        for det in detections:
            cls_name = det["cls_name"]
            conf = det["conf"]
            xyxy = det["xyxy"]
            area = det["area"]

            # 轨枕处理（class_0）
            if cls_name == "class_0":
                area_ratio = area / max_sleeper_area if max_sleeper_area > 0 else 0.0
                if area >= sleeper_area_threshold:
                    valid_sleeper_box_count += 1
                    draw_box(show_img, xyxy, f"sleeper {conf:.2f}", (39, 174, 96))
                else:
                    draw_box(show_img, xyxy, f"sleeper_small {area_ratio:.2f}", (127, 140, 141))

            # 喷号处理（class_1）
            elif cls_name == "class_1":
                draw_box(show_img, xyxy, f"spray {conf:.2f}", (52, 152, 219))
                crop = safe_crop(frame, xyxy, pad=8)
                raw_ocr_text = self.run_ocr(crop)
                clean_ocr_text = normalize_spray_text(raw_ocr_text)

                if clean_ocr_text:
                    ocr_text_this_frame = clean_ocr_text
                    parsed = parse_spray_number(clean_ocr_text)
                    if parsed is not None:
                        parsed_spray = parsed
                        if parsed.get("direction") is not None:
                            self.current_direction = parsed["direction"]
                        self.current_km = parsed["km"]
                        self.current_count_value = parsed["count_value"]
                        self.total_count_value = parsed["count_value"]
                        self.last_valid_spray = parsed["full_text"]
                        self.log_signal.emit(
                            f"有效喷号：{parsed['full_text']}，位置修正为："
                            f"{self.current_direction} {self.current_section} "
                            f"{parsed['km_text']}+{self.current_count_value:04d}"
                        )

            else:
                # 其他未知类别
                draw_box(show_img, xyxy, f"{cls_name} {conf:.2f}", (149, 165, 166))

        # 轨枕计数逻辑
        frame_count_value = valid_sleeper_box_count // 2
        if parsed_spray is None:
            self.total_count_value += frame_count_value
            self.current_count_value += frame_count_value

        self.last_ocr_text = ocr_text_this_frame or self.last_ocr_text

        # 生成记录
        record_type_parts = ["轨枕计数"]
        if parsed_spray is not None:
            record_type_parts.append("喷号修正")

        conf_text = "-"
        spray_text = parsed_spray["full_text"] if parsed_spray is not None else "-"
        count_text = (
            f"有效{valid_sleeper_box_count} / 本帧{frame_count_value} / "
            f"累计{self.current_count_value:04d} / 喷号{spray_text}"
        )

        self.record_signal.emit([
            current_timestamp_ms(), image_name, "+".join(record_type_parts), conf_text,
            self.current_direction, self.current_section, f"K{self.current_km:02d}", count_text
        ])

        # 叠加显示文本
        dir_for_overlay = "UP" if self.current_direction == "上行" else "DOWN" if self.current_direction == "下行" else "UNKNOWN"
        section_for_overlay = to_overlay_ascii(self.current_section)
        spray_for_overlay = parsed_spray["full_text"] if parsed_spray is not None else self.last_valid_spray

        overlay_text = (
            f"Count:{self.total_count_value:04d} | ValidSleeper:{valid_sleeper_box_count}/{raw_sleeper_box_count} | "
            f"Loc:{dir_for_overlay} {section_for_overlay} K{self.current_km:02d}+{self.current_count_value:04d}"
        )
        if spray_for_overlay:
            overlay_text += f" | Spray:{spray_for_overlay}"

        cv2.rectangle(show_img, (0, 0), (min(show_img.shape[1], 1280), 40), (18, 52, 86), -1)
        cv2.putText(show_img, overlay_text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        # 发送状态信号
        status = {
            "image_name": image_name, "total_count_value": self.total_count_value,
            "raw_sleeper_box_count": raw_sleeper_box_count, "valid_sleeper_box_count": valid_sleeper_box_count,
            "frame_count_value": frame_count_value, "direction": self.current_direction,
            "section": self.current_section, "km": self.current_km, "count_value": self.current_count_value,
            "ocr_text": self.last_ocr_text, "valid_spray": self.last_valid_spray,
            "detect_count": detect_count, "sleeper_area_threshold": sleeper_area_threshold
        }

        self.status_signal.emit(status)
        self.frame_signal.emit(show_img)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("地铁巡检定位系统")
        self.resize(1500, 900)

        self.model_path = ""
        self.source_path = ""
        self.worker = None

        self.init_ui()

    def init_ui(self):
        root = QWidget()
        self.setStyleSheet("""
            QMainWindow { background-color: #eef3f8; }
            QWidget { font-family: Microsoft YaHei, SimSun, Arial; font-size: 13px; color: #1f2933; }
            QGroupBox { background-color: #ffffff; border: 1px solid #c7d7e8; border-radius: 8px; margin-top: 12px; padding: 8px; font-weight: bold; color: #16456d; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; background-color: #eef3f8; }
            QPushButton { background-color: #2f80ed; color: white; border: none; border-radius: 6px; padding: 8px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #1c6dd0; }
            QPushButton:pressed { background-color: #155aa8; }
            QPushButton:disabled { background-color: #9bbce6; color: #eef3f8; }
            QLineEdit, QDoubleSpinBox { background-color: #ffffff; border: 1px solid #b8c7d9; border-radius: 5px; padding: 4px; min-height: 22px; }
            QTableWidget { background-color: #ffffff; alternate-background-color: #f2f7fc; gridline-color: #d6e1ee; border: 1px solid #c7d7e8; }
            QHeaderView::section { background-color: #dbeafe; color: #16456d; padding: 6px; border: 1px solid #c7d7e8; font-weight: bold; }
            QTextEdit { background-color: #0f172a; color: #dbeafe; border-radius: 6px; padding: 6px; }
        """)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧布局
        left_layout = QVBoxLayout()
        self.image_label = QLabel("请选择YOLO权重和图像文件夹")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color:#111827; color:#e5f0ff; font-size:20px; border: 2px solid #2f80ed; border-radius: 10px;")
        self.image_label.setMinimumSize(880, 640)

        btn_layout = QGridLayout()
        self.btn_model = QPushButton("选择YOLO权重")
        self.btn_source = QPushButton("选择图像文件夹")
        self.btn_start = QPushButton("开始检测")
        self.btn_stop = QPushButton("停止")
        self.btn_reset = QPushButton("清空表格")
        self.btn_export = QPushButton("导出CSV")

        self.btn_model.clicked.connect(self.choose_model)
        self.btn_source.clicked.connect(self.choose_source)
        self.btn_start.clicked.connect(self.start_detect)
        self.btn_stop.clicked.connect(self.stop_detect)
        self.btn_reset.clicked.connect(self.clear_records)
        self.btn_export.clicked.connect(self.export_csv)

        btn_layout.addWidget(self.btn_model, 0, 0)
        btn_layout.addWidget(self.btn_source, 0, 1)
        btn_layout.addWidget(self.btn_start, 1, 0)
        btn_layout.addWidget(self.btn_stop, 1, 1)
        btn_layout.addWidget(self.btn_reset, 2, 0)
        btn_layout.addWidget(self.btn_export, 2, 1)

        self.path_label = QLabel("模型：未选择\n输入：未选择")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("background:#ffffff; border:1px solid #c7d7e8; border-radius:6px; padding:6px; color:#425466;")

        left_layout.addWidget(self.image_label)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.path_label)

        # 右侧布局
        right_layout = QVBoxLayout()

        tcms_group = QGroupBox("TCMS/MVB区间信息")
        tcms_layout = QGridLayout()
        self.edit_start_station = QLineEdit("A站")
        self.edit_end_station = QLineEdit("D站")
        self.edit_current_station = QLineEdit("A站")
        self.edit_next_station = QLineEdit("B站")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.50)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 0.99)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.50)

        tcms_layout.addWidget(QLabel("起始站"), 0, 0)
        tcms_layout.addWidget(self.edit_start_station, 0, 1)
        tcms_layout.addWidget(QLabel("终点站"), 1, 0)
        tcms_layout.addWidget(self.edit_end_station, 1, 1)
        tcms_layout.addWidget(QLabel("当前站"), 2, 0)
        tcms_layout.addWidget(self.edit_current_station, 2, 1)
        tcms_layout.addWidget(QLabel("下一站"), 3, 0)
        tcms_layout.addWidget(self.edit_next_station, 3, 1)
        tcms_layout.addWidget(QLabel("YOLO置信度"), 4, 0)
        tcms_layout.addWidget(self.spin_conf, 4, 1)
        tcms_layout.addWidget(QLabel("NMS IoU"), 5, 0)
        tcms_layout.addWidget(self.spin_iou, 5, 1)
        tcms_group.setLayout(tcms_layout)

        status_group = QGroupBox("定位状态")
        status_layout = QGridLayout()
        self.lab_total = QLabel("0000")
        self.lab_raw_sleeper = QLabel("0 / 0")
        self.lab_frame_count = QLabel("0")
        self.lab_direction = QLabel("未识别")
        self.lab_section = QLabel("A站-B站")
        self.lab_km = QLabel("K00")
        self.lab_count_value = QLabel("0000")
        self.lab_ocr = QLabel("")
        self.lab_spray = QLabel("")
        self.lab_detect = QLabel("")

        labels = [
            ("累计计数值", self.lab_total),
            ("有效/检测sleeper数", self.lab_raw_sleeper),
            ("当前帧计数值", self.lab_frame_count),
            ("行车方向", self.lab_direction),
            ("站间区间", self.lab_section),
            ("公里标", self.lab_km),
            ("修正计数值", self.lab_count_value),
            ("OCR结果", self.lab_ocr),
            ("有效喷号", self.lab_spray),
            ("检测统计", self.lab_detect),
        ]

        for i, (name, widget) in enumerate(labels):
            name_label = QLabel(name)
            name_label.setStyleSheet("color:#496579;")
            status_layout.addWidget(name_label, i, 0)
            widget.setWordWrap(True)
            widget.setStyleSheet("background:#f8fbff; border:1px solid #d6e1ee; border-radius:4px; padding:3px; color:#0f3b5f;")
            status_layout.addWidget(widget, i, 1)
        status_group.setLayout(status_layout)

        table_group = QGroupBox("巡检定位记录")
        table_layout = QVBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "时间戳", "图像", "记录类型", "置信度", "行车方向", "站间区间", "公里标", "计数值"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.table)
        table_group.setLayout(table_layout)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)

        right_layout.addWidget(tcms_group)
        right_layout.addWidget(status_group)
        right_layout.addWidget(table_group)
        right_layout.addWidget(log_group)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)
        self.setCentralWidget(root)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO权重文件", "", "YOLO Weights (*.pt *.onnx *.engine);;All Files (*)")
        if path:
            self.model_path = path
            self.update_path_label()

    def choose_source(self):
        path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if path:
            self.source_path = path
            self.update_path_label()

    def update_path_label(self):
        self.path_label.setText(f"模型：{self.model_path or '未选择'}\n输入文件夹：{self.source_path or '未选择'}")

    def start_detect(self):
        if not self.model_path:
            QMessageBox.warning(self, "提示", "请先选择YOLO权重文件")
            return
        if not self.source_path:
            QMessageBox.warning(self, "提示", "请先选择图像文件夹")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "提示", "检测正在运行")
            return

        self.table.setRowCount(0)
        self.log_edit.clear()
        self.btn_start.setEnabled(False)

        tcms_info = {
            "start_station": self.edit_start_station.text().strip() or "A站",
            "end_station": self.edit_end_station.text().strip() or "D站",
            "current_station": self.edit_current_station.text().strip() or "A站",
            "next_station": self.edit_next_station.text().strip() or "B站",
        }

        self.worker = DetectionWorker(
            model_path=self.model_path, source_path=self.source_path,
            conf_thres=float(self.spin_conf.value()), iou_thres=float(self.spin_iou.value()),
            tcms_info=tcms_info
        )
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.status_signal.connect(self.update_status)
        self.worker.record_signal.connect(self.add_record)
        self.worker.log_signal.connect(self.add_log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()
        self.add_log("开始检测")

    def stop_detect(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.add_log("正在停止检测")

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.add_log("检测结束")

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def update_status(self, status):
        self.lab_total.setText(f"{int(status.get('total_count_value', 0)):04d}")
        valid_num = int(status.get("valid_sleeper_box_count", 0))
        raw_num = int(status.get("raw_sleeper_box_count", 0))
        self.lab_raw_sleeper.setText(f"{valid_num} / {raw_num}")
        self.lab_frame_count.setText(str(status.get("frame_count_value", 0)))
        self.lab_direction.setText(status.get("direction", ""))
        self.lab_section.setText(status.get("section", ""))
        self.lab_km.setText(f"K{int(status.get('km', 0)):02d}")
        self.lab_count_value.setText(f"{int(status.get('count_value', 0)):04d}")
        self.lab_ocr.setText(status.get("ocr_text", ""))
        self.lab_spray.setText(status.get("valid_spray", ""))
        detect_count = status.get("detect_count", {})
        self.lab_detect.setText(", ".join([f"{k}:{v}" for k, v in detect_count.items()]))

    def add_record(self, row):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, value in enumerate(row):
            self.table.setItem(r, c, QTableWidgetItem(str(value)))
        self.table.scrollToBottom()

    def add_log(self, text):
        now = current_timestamp_ms()
        self.log_edit.append(f"[{now}] {text}")

    def clear_records(self):
        self.table.setRowCount(0)
        self.log_edit.clear()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "inspection_records.csv", "CSV Files (*.csv)")
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
                writer.writerow(headers)
                for r in range(self.table.rowCount()):
                    row = [self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(self.table.columnCount())]
                    writer.writerow(row)
            QMessageBox.information(self, "完成", f"已导出：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{e}")

    def closeEvent(self, event):
        self.stop_detect()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())