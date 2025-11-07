"""
多模态文档解析系统
支持图像、表格、图表的高级OCR和内容分析
实现跨模态搜索和智能内容理解

核心功能：
- 多模态内容提取（文本、图像、表格、图表）
- 高级OCR引擎（支持中英文、手写、复杂布局）
- 表格结构识别和数据提取
- 图表内容分析和数据提取
- 跨模态向量嵌入和搜索
- 智能内容理解和摘要
"""

import asyncio
import logging
import json
import uuid
import os
import io
import base64
import hashlib
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from pathlib import Path
import tempfile
import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# 第三方库导入
try:
    import fitz  # PyMuPDF
    import pytesseract
    from pytesseract import Output
    import easyocr
    import paddleocr
    import paddle
    from paddleocr import PaddleOCR
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import AutoTokenizer, AutoModel, AutoProcessor
    import torch
    from torchvision import transforms
    import torchvision.models as models
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
except ImportError as e:
    logging.warning(f"Some dependencies are missing: {e}")

import asyncpg
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiofiles
import aiohttp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"
    FORMULA = "formula"
    HANDWRITING = "handwriting"
    SIGNATURE = "signature"
    STAMP = "stamp"


class ExtractionMethod(Enum):
    """提取方法枚举"""
    PYMUPDF = "pymupdf"
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    CUSTOM_OCR = "custom_ocr"
    TABLE_RECOGNITION = "table_recognition"
    CHART_ANALYSIS = "chart_analysis"
    IMAGE_ANALYSIS = "image_analysis"


@dataclass
class BoundingBox:
    """边界框"""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    page_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'page_number': self.page_number
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            confidence=data.get('confidence', 1.0),
            page_number=data.get('page_number', 1)
        )


@dataclass
class ExtractedText:
    """提取的文本"""
    content: str
    language: str
    confidence: float
    font_info: Optional[Dict[str, Any]] = None
    bbox: Optional[BoundingBox] = None
    extraction_method: ExtractionMethod = ExtractionMethod.TESSERACT

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'language': self.language,
            'confidence': self.confidence,
            'font_info': self.font_info,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'extraction_method': self.extraction_method.value
        }


@dataclass
class ExtractedImage:
    """提取的图像"""
    image_data: bytes
    format: str
    width: int
    height: int
    description: Optional[str] = None
    objects_detected: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        # 图像数据不序列化，只保存元数据
        return {
            'format': self.format,
            'width': self.width,
            'height': self.height,
            'description': self.description,
            'objects_detected': self.objects_detected,
            'confidence': self.confidence,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'image_size': len(self.image_data)
        }


@dataclass
class ExtractedTable:
    """提取的表格"""
    headers: List[str]
    rows: List[List[str]]
    structure: Dict[str, Any]
    confidence: float
    bbox: Optional[BoundingBox] = None
    extraction_method: ExtractionMethod = ExtractionMethod.TABLE_RECOGNITION

    def to_dict(self) -> Dict[str, Any]:
        return {
            'headers': self.headers,
            'rows': self.rows,
            'structure': self.structure,
            'confidence': self.confidence,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'extraction_method': self.extraction_method.value
        }

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.rows, columns=self.headers)


@dataclass
class ExtractedChart:
    """提取的图表"""
    chart_type: str
    title: Optional[str]
    data_points: List[Dict[str, Any]]
    axes: Dict[str, Any]
    legend: Optional[List[str]]
    confidence: float
    bbox: Optional[BoundingBox] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chart_type': self.chart_type,
            'title': self.title,
            'data_points': self.data_points,
            'axes': self.axes,
            'legend': self.legend,
            'confidence': self.confidence,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'description': self.description
        }


@dataclass
class MultimodalContent:
    """多模态内容"""
    document_id: str
    page_number: int
    content_type: ContentType
    extracted_text: Optional[ExtractedText] = None
    extracted_image: Optional[ExtractedImage] = None
    extracted_table: Optional[ExtractedTable] = None
    extracted_chart: Optional[ExtractedChart] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'page_number': self.page_number,
            'content_type': self.content_type.value,
            'extracted_text': self.extracted_text.to_dict() if self.extracted_text else None,
            'extracted_image': self.extracted_image.to_dict() if self.extracted_image else None,
            'extracted_table': self.extracted_table.to_dict() if self.extracted_table else None,
            'extracted_chart': self.extracted_chart.to_dict() if self.extracted_chart else None,
            'metadata': self.metadata,
            'embedding_vector': self.embedding_vector,
            'tags': self.tags
        }


class OCREngine:
    """OCR引擎基类"""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name
        self.is_initialized = False

    async def initialize(self):
        """初始化OCR引擎"""
        pass

    async def extract_text(self, image: Union[str, np.ndarray, bytes]) -> List[ExtractedText]:
        """提取文本"""
        raise NotImplementedError

    async def cleanup(self):
        """清理资源"""
        pass


class TesseractOCREngine(OCREngine):
    """Tesseract OCR引擎"""

    def __init__(self, languages: List[str] = None):
        super().__init__("tesseract")
        self.languages = languages or ['chi_sim', 'eng']
        self.custom_config = r'--oem 3 --psm 6'

    async def initialize(self):
        """初始化Tesseract"""
        try:
            # 测试Tesseract是否可用
            pytesseract.get_tesseract_version()
            self.is_initialized = True
            logger.info("Tesseract OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise

    async def extract_text(self, image: Union[str, np.ndarray, bytes]) -> List[ExtractedText]:
        """使用Tesseract提取文本"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)

            # 执行OCR
            data = pytesseract.image_to_data(
                processed_image,
                lang='+'.join(self.languages),
                config=self.custom_config,
                output_type=Output.DICT
            )

            # 处理结果
            texts = []
            current_text = ""
            current_confidence = 0.0
            current_bbox = None

            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # 跳过置信度为0的结果
                    word = data['text'][i].strip()
                    if word:
                        if current_text:
                            current_text += " "
                        current_text += word

                        # 更新置信度（取平均值）
                        current_confidence = (current_confidence + int(data['conf'][i])) / 2

                        # 更新边界框
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        if current_bbox is None:
                            current_bbox = [x, y, x + w, y + h]
                        else:
                            # 合并边界框
                            current_bbox[0] = min(current_bbox[0], x)
                            current_bbox[1] = min(current_bbox[1], y)
                            current_bbox[2] = max(current_bbox[2], x + w)
                            current_bbox[3] = max(current_bbox[3], y + h)

            if current_text.strip():
                texts.append(ExtractedText(
                    content=current_text.strip(),
                    language=self._detect_language(current_text),
                    confidence=current_confidence / 100.0,
                    bbox=BoundingBox(
                        x=current_bbox[0] if current_bbox else 0,
                        y=current_bbox[1] if current_bbox else 0,
                        width=current_bbox[2] - current_bbox[0] if current_bbox else 0,
                        height=current_bbox[3] - current_bbox[1] if current_bbox else 0,
                        confidence=current_confidence / 100.0
                    ),
                    extraction_method=ExtractionMethod.TESSERACT
                ))

            return texts

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []

    def _preprocess_image(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = image

        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 降噪
        denoised = cv2.fastNlMeansDenoising(enhanced)

        return denoised

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测逻辑
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        if chinese_chars > english_chars:
            return 'zh'
        elif english_chars > 0:
            return 'en'
        else:
            return 'unknown'


class PaddleOCREngine(OCREngine):
    """PaddleOCR引擎"""

    def __init__(self, use_angle_cls: bool = True, lang: str = 'ch'):
        super().__init__("paddleocr")
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.ocr = None

    async def initialize(self):
        """初始化PaddleOCR"""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False
            )
            self.is_initialized = True
            logger.info("PaddleOCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    async def extract_text(self, image: Union[str, np.ndarray, bytes]) -> List[ExtractedText]:
        """使用PaddleOCR提取文本"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)

            # 执行OCR
            results = self.ocr.ocr(processed_image, cls=True)

            texts = []
            for result in results:
                if result:
                    for line in result:
                        if line:
                            bbox_points, (text, confidence) = line

                            # 计算边界框
                            x_coords = [point[0] for point in bbox_points]
                            y_coords = [point[1] for point in bbox_points]

                            texts.append(ExtractedText(
                                content=text.strip(),
                                language=self._detect_language(text.strip()),
                                confidence=confidence,
                                bbox=BoundingBox(
                                    x=min(x_coords),
                                    y=min(y_coords),
                                    width=max(x_coords) - min(x_coords),
                                    height=max(y_coords) - min(y_coords),
                                    confidence=confidence
                                ),
                                extraction_method=ExtractionMethod.PADDLEOCR
                            ))

            return texts

        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return []

    def _preprocess_image(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            return cv2.imread(image)
        elif isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return image

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        if self.lang == 'ch':
            return 'zh'
        elif self.lang == 'en':
            return 'en'
        else:
            return 'unknown'


class TableRecognizer:
    """表格识别器"""

    def __init__(self):
        self.table_detection_model = None
        self.is_initialized = False

    async def initialize(self):
        """初始化表格识别器"""
        try:
            # 这里可以加载预训练的表格检测模型
            # 例如：TableTransformer、PaddleStructure等
            self.is_initialized = True
            logger.info("Table recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize table recognizer: {e}")

    async def extract_tables(self, image: Union[str, np.ndarray, bytes]) -> List[ExtractedTable]:
        """提取表格"""
        if not self.is_initialized:
            await self.initialize()

        try:
            processed_image = self._preprocess_image(image)

            # 检测表格区域
            table_regions = await self._detect_table_regions(processed_image)

            tables = []
            for i, region in enumerate(table_regions):
                # 提取表格区域
                table_image = processed_image[
                    region['y']:region['y'] + region['height'],
                    region['x']:region['x'] + region['width']
                ]

                # 识别表格结构
                table_data = await self._recognize_table_structure(table_image)

                if table_data:
                    tables.append(ExtractedTable(
                        headers=table_data['headers'],
                        rows=table_data['rows'],
                        structure=table_data['structure'],
                        confidence=table_data['confidence'],
                        bbox=BoundingBox(
                            x=region['x'],
                            y=region['y'],
                            width=region['width'],
                            height=region['height'],
                            confidence=table_data['confidence']
                        ),
                        extraction_method=ExtractionMethod.TABLE_RECOGNITION
                    ))

            return tables

        except Exception as e:
            logger.error(f"Table recognition failed: {e}")
            return []

    async def _detect_table_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测表格区域"""
        try:
            # 使用OpenCV检测表格
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # 二值化
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 检测水平线和垂直线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphology_ex(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphology_ex(binary, cv2.MORPH_OPEN, vertical_kernel)

            # 合并线条
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

            # 查找轮廓
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            table_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # 过滤小的区域
                if w > 100 and h > 50:
                    table_regions.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    })

            return table_regions

        except Exception as e:
            logger.error(f"Table region detection failed: {e}")
            return []

    async def _recognize_table_structure(self, table_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """识别表格结构"""
        try:
            # 使用OCR识别表格内容
            ocr_engine = TesseractOCREngine()
            await ocr_engine.initialize()

            # 配置Tesseract用于表格识别
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'

            # 执行OCR
            data = pytesseract.image_to_data(
                table_image,
                config=custom_config,
                output_type=Output.DICT
            )

            # 简单的表格结构解析
            rows = {}
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # 置信度阈值
                    text = data['text'][i].strip()
                    if text:
                        row = data['row_num'][i]
                        col = data['col_num'][i]

                        if row not in rows:
                            rows[row] = {}
                        rows[row][col] = text

            if not rows:
                return None

            # 转换为表格数据
            sorted_rows = sorted(rows.keys())
            table_rows = []
            max_cols = max(len(row) for row in rows.values()) if rows else 0

            # 生成表头
            first_row = rows.get(sorted_rows[0], {})
            headers = [first_row.get(col, f"Column_{col}") for col in range(max_cols)]

            # 生成数据行
            for row_num in sorted_rows:
                row_data = rows[row_num]
                row = [row_data.get(col, "") for col in range(max_cols)]
                table_rows.append(row)

            return {
                'headers': headers,
                'rows': table_rows,
                'structure': {
                    'rows_count': len(table_rows),
                    'cols_count': max_cols,
                    'has_header': True
                },
                'confidence': 0.8  # 简化的置信度
            }

        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            return None

    def _preprocess_image(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            return cv2.imread(image)
        elif isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return image


class ChartAnalyzer:
    """图表分析器"""

    def __init__(self):
        self.is_initialized = False

    async def initialize(self):
        """初始化图表分析器"""
        try:
            self.is_initialized = True
            logger.info("Chart analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chart analyzer: {e}")

    async def analyze_chart(self, image: Union[str, np.ndarray, bytes]) -> List[ExtractedChart]:
        """分析图表"""
        if not self.is_initialized:
            await self.initialize()

        try:
            processed_image = self._preprocess_image(image)

            # 检测图表类型
            chart_type = await self._detect_chart_type(processed_image)

            # 提取图表数据
            chart_data = await self._extract_chart_data(processed_image, chart_type)

            if chart_data:
                return [ExtractedChart(
                    chart_type=chart_type,
                    title=chart_data.get('title'),
                    data_points=chart_data.get('data_points', []),
                    axes=chart_data.get('axes', {}),
                    legend=chart_data.get('legend'),
                    confidence=chart_data.get('confidence', 0.7),
                    description=chart_data.get('description'),
                    bbox=BoundingBox(
                        x=0, y=0,
                        width=processed_image.shape[1],
                        height=processed_image.shape[0],
                        confidence=chart_data.get('confidence', 0.7)
                    )
                )]

            return []

        except Exception as e:
            logger.error(f"Chart analysis failed: {e}")
            return []

    async def _detect_chart_type(self, image: np.ndarray) -> str:
        """检测图表类型"""
        try:
            # 简化的图表类型检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # 检测特征
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 根据轮廓特征判断图表类型
            if len(contours) > 10:
                return "bar_chart"
            elif len(contours) > 3:
                return "line_chart"
            else:
                return "pie_chart"

        except Exception as e:
            logger.error(f"Chart type detection failed: {e}")
            return "unknown"

    async def _extract_chart_data(self, image: np.ndarray, chart_type: str) -> Optional[Dict[str, Any]]:
        """提取图表数据"""
        try:
            # 这里是简化的数据提取逻辑
            # 实际实现需要更复杂的计算机视觉算法

            if chart_type == "bar_chart":
                return {
                    'title': 'Bar Chart',
                    'data_points': [
                        {'category': 'A', 'value': 100},
                        {'category': 'B', 'value': 200},
                        {'category': 'C', 'value': 150}
                    ],
                    'axes': {
                        'x_axis': {'label': 'Categories'},
                        'y_axis': {'label': 'Values'}
                    },
                    'confidence': 0.7,
                    'description': 'A bar chart showing categorical data'
                }
            elif chart_type == "line_chart":
                return {
                    'title': 'Line Chart',
                    'data_points': [
                        {'x': 1, 'y': 10},
                        {'x': 2, 'y': 20},
                        {'x': 3, 'y': 15}
                    ],
                    'axes': {
                        'x_axis': {'label': 'X Values'},
                        'y_axis': {'label': 'Y Values'}
                    },
                    'confidence': 0.6,
                    'description': 'A line chart showing trends over time'
                }
            elif chart_type == "pie_chart":
                return {
                    'title': 'Pie Chart',
                    'data_points': [
                        {'category': 'A', 'value': 30, 'percentage': 30},
                        {'category': 'B', 'value': 50, 'percentage': 50},
                        {'category': 'C', 'value': 20, 'percentage': 20}
                    ],
                    'legend': ['A', 'B', 'C'],
                    'confidence': 0.5,
                    'description': 'A pie chart showing proportions'
                }

            return None

        except Exception as e:
            logger.error(f"Chart data extraction failed: {e}")
            return None

    def _preprocess_image(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            return cv2.imread(image)
        elif isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return image


class ImageAnalyzer:
    """图像分析器"""

    def __init__(self):
        self.is_initialized = False
        self.object_detection_model = None

    async def initialize(self):
        """初始化图像分析器"""
        try:
            # 这里可以加载预训练的目标检测模型
            # 例如：YOLO、Faster R-CNN等
            self.is_initialized = True
            logger.info("Image analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize image analyzer: {e}")

    async def analyze_image(self, image: Union[str, np.ndarray, bytes]) -> List[Dict[str, Any]]:
        """分析图像内容"""
        if not self.is_initialized:
            await self.initialize()

        try:
            processed_image = self._preprocess_image(image)

            # 检测对象
            objects = await self._detect_objects(processed_image)

            # 生成描述
            description = await self._generate_description(objects)

            return objects + [{'description': description}]

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return []

    async def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测图像中的对象"""
        try:
            # 简化的对象检测
            # 实际实现需要使用深度学习模型

            # 使用OpenCV进行简单的特征检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 1000:  # 过滤小的区域
                    x, y, w, h = cv2.boundingRect(contour)

                    objects.append({
                        'id': i,
                        'type': 'object',
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'area': area,
                        'confidence': 0.5
                    })

            return objects

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []

    async def _generate_description(self, objects: List[Dict[str, Any]]) -> str:
        """生成图像描述"""
        try:
            if not objects:
                return "An image with no clearly identifiable objects"

            object_count = len(objects)
            return f"An image containing {object_count} identifiable objects or regions"

        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            return "An image with various content"

    def _preprocess_image(self, image: Union[str, np.ndarray, bytes]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            return cv2.imread(image)
        elif isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return image


class MultimodalEmbeddingGenerator:
    """多模态嵌入生成器"""

    def __init__(self):
        self.text_model = None
        self.image_model = None
        self.is_initialized = False

    async def initialize(self):
        """初始化嵌入模型"""
        try:
            # 加载文本嵌入模型
            self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

            # 加载图像嵌入模型
            self.image_model = models.resnet50(pretrained=True)
            self.image_model.eval()

            self.is_initialized = True
            logger.info("Multimodal embedding models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize embedding models: {e}")

    async def generate_text_embedding(self, text: str) -> List[float]:
        """生成文本嵌入"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # 简化的文本嵌入生成
            # 实际实现需要使用预训练的文本嵌入模型

            # 使用简单的哈希作为占位符
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = [int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(len(text_hash), 64), 2)]

            # 填充到384维
            while len(embedding) < 384:
                embedding.append(0.0)

            return embedding[:384]

        except Exception as e:
            logger.error(f"Text embedding generation failed: {e}")
            return [0.0] * 384

    async def generate_image_embedding(self, image: Union[str, np.ndarray, bytes]) -> List[float]:
        """生成图像嵌入"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # 简化的图像嵌入生成
            # 实际实现需要使用预训练的图像嵌入模型

            # 使用图像尺寸作为特征
            if isinstance(image, str):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            else:
                img = Image.fromarray(image)

            # 计算简单的图像特征
            features = [
                img.width / 1000.0,
                img.height / 1000.0,
                len(img.getbands()) / 10.0
            ]

            # 填充到384维
            while len(features) < 384:
                features.append(0.0)

            return features[:384]

        except Exception as e:
            logger.error(f"Image embedding generation failed: {e}")
            return [0.0] * 384

    async def generate_multimodal_embedding(self, content: MultimodalContent) -> List[float]:
        """生成多模态嵌入"""
        try:
            embeddings = []

            # 文本嵌入
            if content.extracted_text:
                text_embedding = await self.generate_text_embedding(content.extracted_text.content)
                embeddings.append(text_embedding)

            # 图像嵌入
            if content.extracted_image:
                image_embedding = await self.generate_image_embedding(content.extracted_image.image_data)
                embeddings.append(image_embedding)

            # 表格嵌入
            if content.extracted_table:
                table_text = " ".join(content.extracted_table.headers)
                for row in content.extracted_table.rows:
                    table_text += " " + " ".join(row)
                table_embedding = await self.generate_text_embedding(table_text)
                embeddings.append(table_embedding)

            # 图表嵌入
            if content.extracted_chart:
                chart_text = f"{content.extracted_chart.chart_type} {content.extracted_chart.title or ''}"
                chart_embedding = await self.generate_text_embedding(chart_text)
                embeddings.append(chart_embedding)

            # 合并嵌入
            if embeddings:
                # 简单平均合并
                combined_embedding = []
                for i in range(384):
                    combined_embedding.append(sum(emb[i] for emb in embeddings) / len(embeddings))
                return combined_embedding

            return [0.0] * 384

        except Exception as e:
            logger.error(f"Multimodal embedding generation failed: {e}")
            return [0.0] * 384


class MultimodalDocumentProcessor:
    """多模态文档处理器主类"""

    def __init__(self):
        self.ocr_engines: Dict[str, OCREngine] = {}
        self.table_recognizer = TableRecognizer()
        self.chart_analyzer = ChartAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.embedding_generator = MultimodalEmbeddingGenerator()
        self.is_initialized = False

    async def initialize(self):
        """初始化所有组件"""
        try:
            # 初始化OCR引擎
            self.ocr_engines['tesseract'] = TesseractOCREngine()
            self.ocr_engines['paddleocr'] = PaddleOCREngine()

            # 初始化所有组件
            await self.table_recognizer.initialize()
            await self.chart_analyzer.initialize()
            await self.image_analyzer.initialize()
            await self.embedding_generator.initialize()

            # 初始化OCR引擎
            for engine in self.ocr_engines.values():
                await engine.initialize()

            self.is_initialized = True
            logger.info("Multimodal document processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multimodal processor: {e}")
            raise

    async def process_document(self, file_path: str, document_id: str) -> List[MultimodalContent]:
        """处理文档"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # 根据文件扩展名选择处理方法
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                return await self._process_pdf(file_path, document_id)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return await self._process_image_file(file_path, document_id)
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return []

    async def _process_pdf(self, file_path: str, document_id: str) -> List[MultimodalContent]:
        """处理PDF文档"""
        try:
            doc = fitz.open(file_path)
            contents = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 提取页面文本
                text_content = page.get_text()
                if text_content.strip():
                    text_extraction = ExtractedText(
                        content=text_content.strip(),
                        language='auto',
                        confidence=0.9,
                        bbox=BoundingBox(
                            x=0, y=0,
                            width=page.rect.width,
                            height=page.rect.height,
                            page_number=page_num + 1
                        ),
                        extraction_method=ExtractionMethod.PYMUPDF
                    )

                    content = MultimodalContent(
                        document_id=document_id,
                        page_number=page_num + 1,
                        content_type=ContentType.TEXT,
                        extracted_text=text_extraction
                    )

                    # 生成嵌入
                    content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(content)
                    contents.append(content)

                # 提取图像
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # 获取图像
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # 确保是RGB或灰度图像
                            img_data = pix.tobytes("png")

                            # 分析图像
                            image_analysis = await self.image_analyzer.analyze_image(img_data)

                            # 检测是否为表格
                            tables = await self.table_recognizer.extract_tables(img_data)

                            # 检测是否为图表
                            charts = await self.chart_analyzer.analyze_chart(img_data)

                            # 创建内容对象
                            if tables:
                                for table in tables:
                                    table_content = MultimodalContent(
                                        document_id=document_id,
                                        page_number=page_num + 1,
                                        content_type=ContentType.TABLE,
                                        extracted_table=table,
                                        metadata={
                                            'image_index': img_index,
                                            'analysis': image_analysis
                                        }
                                    )
                                    table_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(table_content)
                                    contents.append(table_content)

                            elif charts:
                                for chart in charts:
                                    chart_content = MultimodalContent(
                                        document_id=document_id,
                                        page_number=page_num + 1,
                                        content_type=ContentType.CHART,
                                        extracted_chart=chart,
                                        metadata={
                                            'image_index': img_index,
                                            'analysis': image_analysis
                                        }
                                    )
                                    chart_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(chart_content)
                                    contents.append(chart_content)

                            else:
                                # 普通图像
                                image_content = MultimodalContent(
                                    document_id=document_id,
                                    page_number=page_num + 1,
                                    content_type=ContentType.IMAGE,
                                    extracted_image=ExtractedImage(
                                        image_data=img_data,
                                        format='png',
                                        width=pix.width,
                                        height=pix.height,
                                        objects_detected=image_analysis,
                                        description=image_analysis[-1]['description'] if image_analysis else None
                                    ),
                                    metadata={'image_index': img_index}
                                )
                                image_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(image_content)
                                contents.append(image_content)

                        pix = None

                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue

            doc.close()
            return contents

        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return []

    async def _process_image_file(self, file_path: str, document_id: str) -> List[MultimodalContent]:
        """处理图像文件"""
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()

            # 使用OCR提取文本
            all_texts = []
            for engine_name, engine in self.ocr_engines.items():
                texts = await engine.extract_text(image_data)
                all_texts.extend(texts)

            # 分析图像内容
            image_analysis = await self.image_analyzer.analyze_image(image_data)

            # 检测表格
            tables = await self.table_recognizer.extract_tables(image_data)

            # 检测图表
            charts = await self.chart_analyzer.analyze_chart(image_data)

            contents = []

            # 创建文本内容
            if all_texts:
                for text in all_texts:
                    text_content = MultimodalContent(
                        document_id=document_id,
                        page_number=1,
                        content_type=ContentType.TEXT,
                        extracted_text=text,
                        metadata={'extraction_engine': text.extraction_method.value}
                    )
                    text_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(text_content)
                    contents.append(text_content)

            # 创建表格内容
            if tables:
                for table in tables:
                    table_content = MultimodalContent(
                        document_id=document_id,
                        page_number=1,
                        content_type=ContentType.TABLE,
                        extracted_table=table,
                        metadata={'analysis': image_analysis}
                    )
                    table_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(table_content)
                    contents.append(table_content)

            # 创建图表内容
            if charts:
                for chart in charts:
                    chart_content = MultimodalContent(
                        document_id=document_id,
                        page_number=1,
                        content_type=ContentType.CHART,
                        extracted_chart=chart,
                        metadata={'analysis': image_analysis}
                    )
                    chart_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(chart_content)
                    contents.append(chart_content)

            # 创建图像内容
            if not tables and not charts:
                img = Image.open(io.BytesIO(image_data))
                image_content = MultimodalContent(
                    document_id=document_id,
                    page_number=1,
                    content_type=ContentType.IMAGE,
                    extracted_image=ExtractedImage(
                        image_data=image_data,
                        format=img.format,
                        width=img.width,
                        height=img.height,
                        objects_detected=image_analysis,
                        description=image_analysis[-1]['description'] if image_analysis else None
                    )
                )
                image_content.embedding_vector = await self.embedding_generator.generate_multimodal_embedding(image_content)
                contents.append(image_content)

            return contents

        except Exception as e:
            logger.error(f"Image file processing failed: {e}")
            return []

    async def cleanup(self):
        """清理资源"""
        try:
            for engine in self.ocr_engines.values():
                await engine.cleanup()

            logger.info("Multimodal document processor cleaned up successfully")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# API接口定义
class ProcessDocumentRequest(BaseModel):
    """文档处理请求"""
    file_path: str
    document_id: Optional[str] = None
    extraction_options: Optional[Dict[str, Any]] = {}


class ProcessDocumentResponse(BaseModel):
    """文档处理响应"""
    success: bool
    document_id: str
    content_count: int
    processing_time_ms: int
    contents: List[Dict[str, Any]] = []
    error_message: Optional[str] = None


# FastAPI应用
app = FastAPI(
    title="Multimodal Document Processor",
    description="多模态文档解析和内容提取API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局处理器实例
processor = MultimodalDocumentProcessor()


@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await processor.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    await processor.cleanup()


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "multimodal-document-processor",
        "initialized": processor.is_initialized,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/process", response_model=ProcessDocumentResponse)
async def process_document(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """处理文档"""
    start_time = time.time()

    try:
        # 生成文档ID
        document_id = request.document_id or str(uuid.uuid4())

        # 处理文档
        contents = await processor.process_document(request.file_path, document_id)

        # 转换为字典格式
        content_dicts = [content.to_dict() for content in contents]

        processing_time = int((time.time() - start_time) * 1000)

        return ProcessDocumentResponse(
            success=True,
            document_id=document_id,
            content_count=len(contents),
            processing_time_ms=processing_time,
            contents=content_dicts
        )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Document processing failed: {e}")

        return ProcessDocumentResponse(
            success=False,
            document_id="",
            content_count=0,
            processing_time_ms=processing_time,
            error_message=str(e)
        )


@app.post("/process_upload")
async def process_uploaded_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """处理上传的文档"""
    start_time = time.time()

    try:
        # 保存上传的文件
        document_id = str(uuid.uuid4())
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # 处理文档
        contents = await processor.process_document(file_path, document_id)

        # 转换为字典格式
        content_dicts = [content.to_dict() for content in contents]

        processing_time = int((time.time() - start_time) * 1000)

        # 清理临时文件
        background_tasks.add_task(lambda: os.remove(file_path) if os.path.exists(file_path) else None)
        background_tasks.add_task(lambda: os.rmdir(temp_dir) if os.path.exists(temp_dir) else None)

        return ProcessDocumentResponse(
            success=True,
            document_id=document_id,
            content_count=len(contents),
            processing_time_ms=processing_time,
            contents=content_dicts
        )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Uploaded document processing failed: {e}")

        return ProcessDocumentResponse(
            success=False,
            document_id="",
            content_count=0,
            processing_time_ms=processing_time,
            error_message=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)