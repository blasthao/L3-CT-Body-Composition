# -*- coding: utf-8 -*-
"""
程序名称：L3 层面解剖结构 AI 自动分割系统 (绿色免安装便携版 + PACS传输功能)
功能描述：支持上传 DICOM ZIP 压缩包或 NIfTI 文件，自动定位 L3 中心层面并提取 4 种解剖特征面积。
          包含一页 A4 规范化临床评估报告生成（PDF）与 二次捕获（SC）DICOM 封装。
          已集成：防弹级编码保护（清除底层\x00空字符）、消除UI警告、PACS 一键归档。
          特化更新：引入中国健康人群体成分分布基准，优化文献参考体系与提示逻辑。
"""

import os
import sys
import io
import glob
import zipfile
import tempfile
import shutil
import logging
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
import torch
import psutil
from datetime import datetime
import streamlit as st
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# 全局静音 SimpleITK/GDCM 的底层 C++ 扫描警告
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# ==============================
# 动态加载高级依赖 (PDF 与 PACS)
# ==============================
try:
    import pydicom
    from pydicom.uid import generate_uid
    from pynetdicom import AE, VerificationPresentationContexts, build_context
    from pynetdicom.sop_class import SecondaryCaptureImageStorage
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    import fitz  # 引入 PyMuPDF，将 PDF 转换为最高兼容性的 SC DICOM 图像
    
    # 注册 ReportLab 内置的免费中文字体
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    HAS_PRO_LIBS = True
except ImportError:
    HAS_PRO_LIBS = False

# ==============================
# 终极日志拦截：屏蔽 Streamlit 多线程运行时的无用警告
# ==============================
warnings.filterwarnings("ignore")
for logger_name in logging.root.manager.loggerDict:
    if "streamlit" in logger_name:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ==============================
# 多语言配置字典 (中、英、日)
# ==============================
LANG = {
    "zh": {
        "title": "L3 层面解剖结构 AI 自动分割系统",
        "subtitle": "支持上传 `DICOM ZIP压缩包` 或 `3D NIfTI` 文件，自动定位 L3 中心层面并提取 4 种解剖特征面积。",
        "sidebar_title": "⚙️ 模型权重配置",
        "hw_monitor": "💻 运行设备状态",
        "pacs_title": "📡 PACS 归档配置",
        "ts_weight": "TotalSegmentator 权重目录",
        "nnunet_weight": "nnU-Net 2D 权重目录",
        "lang_select": "🌐 语言 / Language",
        "upload_title": "### 📂 上传影像数据",
        "upload_hint": "支持上传 .zip (包含DCM序列) 或 .nii.gz 格式文件",
        "height_input": "📏 身高 (cm) [必填]",
        "sex_input": "🚻 性别 [必填]",
        "age_input": "👤 年龄 (岁)",
        "btn_start": "🚀 开始自动化分析",
        "status_parse": "正在解析上传文件...",
        "status_ts": "正在运行 TotalSegmentator 进行 L3 层级定位...",
        "status_nnunet": "正在运行 nnU-Net 2D 进行多分类解剖结构推理...",
        "status_calc": "计算解剖学面积指标并渲染图像...",
        "success_l3": "✅ 成功定位 L3 层级，Z轴中心切片坐标为: ",
        "err_l3": "L3 定位失败: ",
        "err_nnunet": "nnU-Net 推理失败: ",
        "panel_title": "### 📊 分析结果面板 (检验报告视角)",
        "disclaimer": "⚠️ **免责声明**：本系统计算的影像学指标及参考评估均基于特定科研文献队列（详见文末引文），仅供科学研究与临床探索参考，**不具备直接临床诊断价值**。请结合患者实际体征综合评估。",
        "orig_img": "Original L3 Slice",
        "seg_img": "AI Segmentation",
        "table_title": "#### 区域量化指标与临床参考评估",
        "col_anat": "解剖结构",
        "col_area": "横截面积(cm²)",
        "col_hu": "平均密度(HU)",
        "col_index": "测试结果",
        "col_ref": "参考区间 [引文]",
        "col_status": "状态提示",
        "col_muscle_hu": "骨骼肌密度 (Muscle HU)",
        "ref_muscle_hu": "< 41/33 [6]",
        "ref_vatsat": "暂无统一截断值 [5]",
        "status_check_bmi": "结合BMI评估",
        "btn_zip": "📥 导出完整分析数据包 (.zip)",
        "csv_filename": "L3_segmentation_metrics.csv",
        "status_normal": "正常区间",
        "status_low_ref": "低值参考",
        "status_na": "-",
        "labels": {
            1: "松质骨 (Cancellous Bone)",
            2: "皮下脂肪 (SAT)",
            3: "内脏脂肪 (VAT)",
            4: "骨骼肌 (Muscle)"
        },
        "citations_title": "📚 临床参考指标与文献引用摘要",
        "citations_text": """
**[1] 椎体松质骨平均密度 (Cancellous Bone Attenuation)**
* **临床意义**：腹部 CT 椎体小梁骨 CT 衰减值可作为骨密度的机会性影像学指标，用于识别低骨密度及骨质疏松风险。
* **参考值**：目前尚无公认的通用 L3 临床截断值。既往研究提出 L1 椎体小梁骨 ≤ 110 HU 可用于机会性骨质疏松筛查，但该阈值不宜直接外推至 L3。
* **引用文献**：Praveen AD, et al. Osteoporos Int. 2024;35(6):971-996; Pickhardt PJ, et al. Ann Intern Med. 2013;158(8):588-595.

**[2] 皮下脂肪指数 (SATI)**
* **临床意义**：SATI 用于标准化评价皮下脂肪储备及分布特征，其临床意义受年龄、性别、BMI 及基础疾病影响。
* **健康人群参考值**：北方中国健康成年人 L3 层面中位数：男性约 38.8 cm²/m²，女性约 55.8 cm²/m²。
* **注**：上述数值为健康人群参考分布，不作为疾病诊断截断值。
* **引用文献**：Kong M, et al. Front Nutr. 2022;9:871697.

**[3] 内脏脂肪指数 (VATI)**
* **临床意义**：VATI 用于标准化评价内脏脂肪蓄积及腹部脂肪分布特征，并受年龄、性别和 BMI 影响。
* **健康人群参考值**：北方中国健康成年人 L3 层面中位数：男性约 39.1 cm²/m²，女性约 25.6 cm²/m²。
* **注**：上述数值为健康人群参考分布，不作为疾病诊断截断值。
* **引用文献**：Kong M, et al. Front Nutr. 2022;9:871697.

**[4] 骨骼肌指数 (SMI)**
* **临床意义**：SMI 为 L3 骨骼肌横截面积与身高平方之比，用于标准化评估骨骼肌量，广泛应用于肌少症、营养状态及临床预后研究。
* **参考值**：北方中国健康成年人基于第 5 百分位建立的低 L3-SMI 参考值为男性 < 40.2 cm²/m²、女性 < 31.6 cm²/m²。
* **注**：上述参考值主要用于人群研究和低肌肉量评估，不作为所有人群的通用肌少症诊断标准。
* **引用文献**：Kong M, et al. Clin Nutr. 2022;41(2):396-404; Prado CM, et al. Lancet Oncol. 2008;9(7):629-635.

**[5] 内脏与皮下脂肪比值 (VAT/SAT Ratio)**
* **临床意义**：反映腹部脂肪在内脏和皮下脂肪库之间的相对分布。较高水平通常提示脂肪分布更偏向内脏脂肪，并与部分心血管及代谢危险因素相关。
* **参考值**：目前尚无适用于不同人群的统一临床截断值。
* **引用文献**：Emamat H, et al. BMC Public Health. 2024;24:1827.

**[6] 骨骼肌密度 (Muscle Attenuation / Muscle HU)**
* **临床意义**：骨骼肌平均 HU 反映肌肉组织衰减特征。较低 HU 通常与肌肉脂肪浸润增加相关，可作为肌肉质量和肌脂肪变性（Myosteatosis）的影像学指标。
* **参考值**：目前尚无适用于所有人群的统一临床截断值。既往肿瘤研究常采用 BMI 分层参考值：BMI < 25 kg/m² 时 < 41 HU，BMI ≥ 25 kg/m² 时 < 33 HU。
* **注**：上述阈值主要来源于肿瘤人群，不宜直接作为一般人群的肌脂肪变性诊断标准。
* **引用文献**：Garcia-Diez AI, et al. Quant Imaging Med Surg. 2024;14(11):7937-7957; Martin L, et al. J Clin Oncol. 2013;31(12):1539-1547.
        """,
        "pdf": {
            "report_title": "L3 层面解剖结构 AI 自动分割规范报告",
            "patient_info": "姓名: {name} &nbsp;&nbsp;&nbsp; 性别: {sex} &nbsp;&nbsp;&nbsp; 年龄: {age} 岁 &nbsp;&nbsp;&nbsp; 身高: {height} cm",
            "report_time": "报告生成时间: {time}",
            "table_title": "1. 区域量化指标与临床参考评估:",
            "interp_title": "2. 重点指标临床参考与提示:",
            "interp_bone": "椎体松质骨密度低于 110 HU：低于经典 L1 机会性骨质疏松参考线，提示关注骨质流失风险（注：L3 尚无公认通用阈值）。",
            "interp_muscle": "骨骼肌指数偏低：低于北方中国健康人群第 5 百分位参考值，提示低肌肉量风险，建议结合机能测评评估。",
            "interp_ratio": "内脏/皮下脂肪比偏高（≥ 1.0）：提示内脏异位脂肪分布显著积聚，与不良心血管及代谢综合征表型相关。",
            "no_abnormal": "所有量化指标均在一般文献参考分布范围内，未见显著边缘偏离。",
            "disclaimer_title": "3. 免责说明：",
            "disclaimer_text": "本系统计算的影像学指标及参考评估仅供科研与临床探索参考，不作为直接临床诊断依据，请结合患者实际体征综合评估。",
            "ref_title": "4. 临床参考指标与文献：",
            "pdf_ref_headers": ["指标名称", "临床意义及特征", "参考阈值", "引用文献"],
            "pdf_ref_data": [
                [
                    "<b>[1] 椎体松质骨密度</b><br/><font size='7' color='#4B5563'>(Cancellous Bone Atten.)</font>",
                    "小梁骨CT衰减值机会性筛查，用于识别低骨密度及骨折风险。",
                    "<b>L1 ≤ 110 HU</b><br/><font size='7' color='#4B5563'>(L3暂无统一截断值)</font>",
                    "Praveen AD, 2024;<br/>Pickhardt PJ, 2013"
                ],
                [
                    "<b>[2] 皮下脂肪指数</b><br/><font size='7' color='#4B5563'>(SATI)</font>",
                    "标准化评价皮下脂肪储备，受年龄、性别、BMI及基础疾病影响。",
                    "中国健康中位数：<br/>男~38.8, 女~55.8",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[3] 内脏脂肪指数</b><br/><font size='7' color='#4B5563'>(VATI)</font>",
                    "标准化评价内脏脂肪蓄积特征，通常随年龄增长呈上升趋势。",
                    "中国健康中位数：<br/>男~39.1, 女~25.6",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[4] 骨骼肌指数</b><br/><font size='7' color='#4B5563'>(SMI)</font>",
                    "用于标准化评估骨骼肌量，广泛用于肌少症及预后研究。",
                    "第5百分位参考值：<br/><b>男&lt;40.2, 女&lt;31.6</b>",
                    "Kong M, et al. 2022;<br/>Prado CM, et al. 2008"
                ],
                [
                    "<b>[5] 内脏/皮下脂肪比</b><br/><font size='7' color='#4B5563'>(VAT/SAT Ratio)</font>",
                    "反映腹部脂肪在内脏/皮下库相对分布，高值与不良代谢风险相关。",
                    "暂无通用截断值<br/><font size='7' color='#4B5563'>(需结合结局评估)</font>",
                    "Emamat H, et al.<br/>BMC Public Health. 2024"
                ],
                [
                    "<b>[6] 骨骼肌密度</b><br/><font size='7' color='#4B5563'>(Muscle HU)</font>",
                    "反映肌肉组织衰减与脂肪浸润，评估肌脂肪变性（Myosteatosis）。",
                    "BMI&lt;25: <b>&lt; 41 HU</b><br/>BMI≥25: <b>&lt; 33 HU</b>",
                    "Garcia-Diez AI, 2024;<br/>Martin L, et al. 2013"
                ]
            ]
        }
    },
    "en": {
        "title": "L3 Slice Anatomical AI Segmentation System",
        "subtitle": "Supports uploading `DICOM ZIP archives` or `3D NIfTI` files. Automatically locates the L3 center slice and extracts 4 anatomical feature areas.",
        "sidebar_title": "⚙️ Model Weight Configuration",
        "hw_monitor": "💻 Device Status",
        "pacs_title": "📡 PACS Archive Config",
        "ts_weight": "TotalSegmentator Weight Directory",
        "nnunet_weight": "nnU-Net 2D Weight Directory",
        "lang_select": "🌐 Language",
        "upload_title": "### 📂 Upload Imaging Data",
        "upload_hint": "Supports .zip (DICOM sequence) or .nii.gz files",
        "height_input": "📏 Height (cm)",
        "sex_input": "🚻 Gender",
        "age_input": "👤 Age",
        "btn_start": "🚀 Start Automated Analysis",
        "status_parse": "Parsing uploaded file...",
        "status_ts": "Running TotalSegmentator for L3 localization...",
        "status_nnunet": "Running nnU-Net 2D for multi-class inference...",
        "status_calc": "Calculating metrics and rendering images...",
        "success_l3": "✅ Located L3 level, Z-axis slice: ",
        "err_l3": "L3 localization failed: ",
        "err_nnunet": "nnU-Net inference failed: ",
        "panel_title": "### 📊 Clinical Analysis Report",
        "disclaimer": "⚠️ **Disclaimer**: The radiological metrics and reference evaluations calculated by this system are based on specific research cohorts. They are for scientific research and clinical reference only, and **do not constitute direct clinical diagnostic value**.",
        "orig_img": "Original L3 Slice",
        "seg_img": "AI Segmentation",
        "table_title": "#### Regional Metrics & Clinical Assessment",
        "col_anat": "Anatomy",
        "col_area": "Area(cm²)",
        "col_hu": "Density(HU)",
        "col_index": "Result",
        "col_ref": "Ref Range [Ref]",
        "col_status": "Status",
        "col_muscle_hu": "Muscle Density (HU)",
        "ref_muscle_hu": "< 41/33 [6]",
        "ref_vatsat": "No unified cutoff [5]",
        "status_check_bmi": "Check BMI",
        "btn_zip": "📥 Export Complete Package (.zip)",
        "csv_filename": "L3_segmentation_metrics_en.csv",
        "status_normal": "Normal Range",
        "status_low_ref": "Low Ref",
        "status_na": "-",
        "labels": {
            1: "Cancellous Bone",
            2: "Subcutaneous Adipose Tissue (SAT)",
            3: "Visceral Adipose Tissue (VAT)",
            4: "Skeletal Muscle"
        },
        "citations_title": "📚 Clinical References and Citations",
        "citations_text": """
**[1] Cancellous Bone Attenuation**
* **Clinical Significance**: Trabecular bone CT attenuation provides opportunistic imaging biomarkers for low bone mineral density and osteoporosis risk.
* **Reference Values**: No universal L3 clinical cutoff exists. L1 trabecular attenuation ≤ 110 HU is widely cited for opportunistic screening, but cannot be directly generalized to L3.
* **Citations**: Praveen AD, et al. Osteoporos Int. 2024;35(6):971-996; Pickhardt PJ, et al. Ann Intern Med. 2013;158(8):588-595.

**[2] Subcutaneous Adipose Tissue Index (SATI)**
* **Clinical Significance**: Evaluates subcutaneous fat reserves and distribution, influenced by age, sex, BMI, and clinical comorbidities.
* **Reference Values**: Healthy northern Chinese adult medians at L3: Male ~38.8 cm²/m², Female ~55.8 cm²/m². (Descriptive reference, not a diagnostic cutoff).
* **Citation**: Kong M, et al. Front Nutr. 2022;9:871697.

**[3] Visceral Adipose Tissue Index (VATI)**
* **Clinical Significance**: Quantifies visceral fat accumulation, typically increasing with age and BMI.
* **Reference Values**: Healthy northern Chinese adult medians at L3: Male ~39.1 cm²/m², Female ~25.6 cm²/m².
* **Citation**: Kong M, et al. Front Nutr. 2022;9:871697.

**[4] Skeletal Muscle Index (SMI)**
* **Clinical Significance**: Skeletal muscle area normalized by height squared. Standard metric for evaluating muscle mass in sarcopenia and cachexia studies.
* **Reference Values**: 5th percentile cutoffs in healthy northern Chinese adults: Male < 40.2 cm²/m², Female < 31.6 cm²/m². (Population reference, not an absolute clinical diagnostic criterion).
* **Citations**: Kong M, et al. Clin Nutr. 2022;41(2):396-404; Prado CM, et al. Lancet Oncol. 2008;9(7):629-635.

**[5] VAT/SAT Ratio**
* **Clinical Significance**: Reflects relative distribution between visceral and subcutaneous fat depots. Elevated ratios associate with unfavorable cardiometabolic risk profiles.
* **Reference Values**: No universal cutoff exists across diverse populations.
* **Citation**: Emamat H, et al. BMC Public Health. 2024;24:1827.

**[6] Muscle Attenuation / Muscle HU**
* **Clinical Significance**: Mean HU reflects tissue attenuation and lipid infiltration, serving as an imaging biomarker for myosteatosis.
* **Reference Values**: Literature BMI-stratified reference cutoffs: BMI < 25 kg/m²: < 41 HU; BMI ≥ 25 kg/m²: < 33 HU. (Mainly from oncology cohorts).
* **Citations**: Garcia-Diez AI, et al. Quant Imaging Med Surg. 2024;14(11):7937-7957; Martin L, et al. J Clin Oncol. 2013;31(12):1539-1547.
        """,
        "pdf": {
            "report_title": "L3 Level Anatomy AI Segmentation Report",
            "patient_info": "Name: {name} &nbsp;&nbsp;&nbsp; Sex: {sex} &nbsp;&nbsp;&nbsp; Age: {age} y &nbsp;&nbsp;&nbsp; Height: {height} cm",
            "report_time": "Report Date: {time}",
            "table_title": "1. Regional Quantitative Metrics & Clinical Assessment:",
            "interp_title": "2. Clinical Reference Notes & Prompts:",
            "interp_bone": "Cancellous bone attenuation < 110 HU: Below classical L1 opportunistic screening benchmark. Note: no established universal L3 cutoff.",
            "interp_muscle": "Low Skeletal Muscle Index: Below 5th percentile reference in healthy adults, indicating low muscle mass risk.",
            "interp_ratio": "Elevated VAT/SAT Ratio (≥ 1.0): Indicates disproportionate visceral fat accumulation, linked to metabolic risk.",
            "no_abnormal": "All evaluated quantitative metrics fall within standard literature reference ranges.",
            "disclaimer_title": "3. Disclaimer:",
            "disclaimer_text": "Metrics and evaluations are intended for scientific research and clinical reference only, and do not constitute direct diagnostic decisions.",
            "ref_title": "4. Clinical References & Thresholds:",
            "pdf_ref_headers": ["Metric", "Clinical Significance", "Ref. Threshold", "Citation"],
            "pdf_ref_data": [
                [
                    "<b>[1] Cancellous Bone</b><br/><font size='7' color='#4B5563'>(Attenuation)</font>",
                    "Opportunistic screening for low bone density and fracture risk.",
                    "<b>L1 ≤ 110 HU</b><br/><font size='7' color='#4B5563'>(No universal L3 cutoff)</font>",
                    "Praveen AD, 2024;<br/>Pickhardt PJ, 2013"
                ],
                [
                    "<b>[2] SAT Index</b><br/><font size='7' color='#4B5563'>(SATI)</font>",
                    "Normalized evaluation of subcutaneous fat reserve and distribution.",
                    "Healthy medians:<br/>Male~38.8, Female~55.8",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[3] VAT Index</b><br/><font size='7' color='#4B5563'>(VATI)</font>",
                    "Quantifies visceral adipose accumulation, increasing with age.",
                    "Healthy medians:<br/>Male~39.1, Female~25.6",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[4] Muscle Index</b><br/><font size='7' color='#4B5563'>(SMI)</font>",
                    "Standard metric for skeletal muscle mass in sarcopenia research.",
                    "5th percentile:<br/><b>M &lt; 40.2, F &lt; 31.6</b>",
                    "Kong M, et al. 2022;<br/>Prado CM, et al. 2008"
                ],
                [
                    "<b>[5] VAT/SAT Ratio</b>",
                    "Relative visceral vs. subcutaneous distribution; metabolic risk marker.",
                    "No universal cutoff<br/><font size='7' color='#4B5563'>(Cohort-dependent)</font>",
                    "Emamat H, et al.<br/>BMC Public Health. 2024"
                ],
                [
                    "<b>[6] Muscle Density</b><br/><font size='7' color='#4B5563'>(Muscle HU)</font>",
                    "Evaluates myosteatosis and ectopic lipid infiltration.",
                    "BMI&lt;25: <b>&lt; 41 HU</b><br/>BMI≥25: <b>&lt; 33 HU</b>",
                    "Garcia-Diez AI, 2024;<br/>Martin L, et al. 2013"
                ]
            ]
        }
    },
    "ja": {
        "title": "L3スライス解剖学的AI自動セグメンテーションシステム",
        "subtitle": "`DICOM ZIPアーカイブ` または `3D NIfTI` ファイルのアップロードをサポートします。L3中心スライスを自動的に特定し、4つの領域を抽出します。",
        "sidebar_title": "⚙️ モデル重み設定",
        "hw_monitor": "💻 デバイスステータス",
        "pacs_title": "📡 PACS アーカイブ設定",
        "ts_weight": "TotalSegmentator 重みディレクトリ",
        "nnunet_weight": "nnU-Net 2D 重みディレクトリ",
        "lang_select": "🌐 言語 / Language",
        "upload_title": "### 📂 画像データをアップロード",
        "upload_hint": ".zip または .nii.gz ファイルをサポート",
        "height_input": "📏 身長 (cm)",
        "sex_input": "🚻 性别",
        "age_input": "👤 年齢",
        "btn_start": "🚀 自動分析を開始",
        "status_parse": "アップロードされたファイルを解析中...",
        "status_ts": "TotalSegmentatorを起動し、L3レベルを特定中...",
        "status_nnunet": "nnU-Net 2D推論を実行中...",
        "status_calc": "指標を計算し、画像をレンダリング中...",
        "success_l3": "✅ L3レベルの特定に成功しました。Z軸スライス: ",
        "err_l3": "L3の特定に失敗しました: ",
        "err_nnunet": "nnU-Net推論に失敗しました: ",
        "panel_title": "### 📊 臨床分析レポート",
        "disclaimer": "⚠️ **免責事項**：本システムで計算された画像解析指標および評価は特定の研究コホートに基づいています。科学研究および臨床参考用であり、**直接的な臨床診断の価値を有するものではありません**。",
        "orig_img": "Original L3 Slice",
        "seg_img": "AI Segmentation",
        "table_title": "#### 定量指標と臨床評価",
        "col_anat": "解剖学的構造",
        "col_area": "断面積(cm²)",
        "col_hu": "密度(HU)",
        "col_index": "テスト結果",
        "col_ref": "基準範囲 [文献]",
        "col_status": "ステータス",
        "col_muscle_hu": "骨格筋密度 (Muscle HU)",
        "ref_muscle_hu": "< 41/33 [6]",
        "ref_vatsat": "統一基準なし [5]",
        "status_check_bmi": "BMI要確認",
        "btn_zip": "📥 完全なパッケージをエクスポート (.zip)",
        "csv_filename": "L3_segmentation_metrics_ja.csv",
        "status_normal": "正常範囲",
        "status_low_ref": "低値参考",
        "status_na": "-",
        "labels": {
            1: "海綿骨 (Cancellous Bone)",
            2: "皮下脂肪 (SAT)",
            3: "内臓脂肪 (VAT)",
            4: "骨格筋 (Muscle)"
        },
        "citations_title": "📚 臨床基準と参考文献",
        "citations_text": """
**[1] 海綿骨平均密度 (Cancellous Bone Attenuation)**
* **臨床的意義**: 腹部CT椎体小梁骨のCT減弱値は、低骨密度および骨粗鬆症リスクを同定するための機会的画像指標となります。
* **基準値**: 現在、統一されたL3臨床カットオフ値は確立されていません。L1小梁骨≤110 HUが機会的スクリーニングに広く引用されていますが、直接L3に適用することは推奨されません。
* **引用文献**: Praveen AD, et al. Osteoporos Int. 2024;35(6):971-996; Pickhardt PJ, et al. Ann Intern Med. 2013;158(8):588-595.

**[2] 皮下脂肪指数 (SATI)**
* **臨床的意義**: 皮下脂肪の蓄積と分布を標準化評価する指標。年齢、性別、BMIの影響を受けます。
* **健康人群基準値**: 中国北方健康成人L3中央値：男性約38.8 cm²/m²、女性約55.8 cm²/m²（参考分布であり診断基準ではありません）。
* **引用文献**: Kong M, et al. Front Nutr. 2022;9:871697.

**[3] 内臓脂肪指数 (VATI)**
* **臨床的意義**: 内臓脂肪の蓄積を標準化評価する指標。通常、年齢とともに増加傾向を示します。
* **健康人群基準値**: 中国北方健康成人L3中央値：男性約39.1 cm²/m²、女性約25.6 cm²/m²。
* **引用文献**: Kong M, et al. Front Nutr. 2022;9:871697.

**[4] 骨格筋指数 (SMI)**
* **臨床的意義**: 骨格筋量を標準化評価する指標。サルコペニアや悪液質の評価に広く利用されています。
* **基準値**: 中国北方健康成人の第5パーセンタイル基準値：男性 < 40.2 cm²/m²、女性 < 31.6 cm²/m²。
* **引用文献**: Kong M, et al. Clin Nutr. 2022;41(2):396-404; Prado CM, et al. Lancet Oncol. 2008;9(7):629-635.

**[5] 内臓/皮下脂肪比 (VAT/SAT Ratio)**
* **臨床的意義**: 内臓脂肪と皮下脂肪の相対的分布を反映。高い値は心血管代謝リスクと関連します。
* **基準値**: 多様な集団に適用可能な統一カットオフ値は現在ありません。
* **引用文献**: Emamat H, et al. BMC Public Health. 2024;24:1827.

**[6] 骨格筋密度 (Muscle Attenuation / Muscle HU)**
* **臨床的意義**: 筋肉内の異所性脂肪沈着（筋脂肪化・Myosteatosis）を反映します。
* **基準値**: 文献的BMI層別参考カットオフ：BMI < 25 kg/m²で < 41 HU、BMI ≥ 25 kg/m²で < 33 HU。
* **引用文献**: Garcia-Diez AI, et al. Quant Imaging Med Surg. 2024;14(11):7937-7957; Martin L, et al. J Clin Oncol. 2013;31(12):1539-1547.
        """,
        "pdf": {
            "report_title": "L3スライス解剖学的AIセグメンテーション報告書",
            "patient_info": "氏名: {name} &nbsp;&nbsp;&nbsp; 性別: {sex} &nbsp;&nbsp;&nbsp; 年齢: {age} 歳 &nbsp;&nbsp;&nbsp; 身長: {height} cm",
            "report_time": "作成日時: {time}",
            "table_title": "1. 領域定量指標および臨床評価:",
            "interp_title": "2. 重点指標の臨床的解釈・注記:",
            "interp_bone": "海綿骨密度が 110 HU 未満: 古典的L1機会的スクリーニング基準を下回っています（注：L3統一基準は未確立）。",
            "interp_muscle": "骨格筋指数低値: 健康成人第5パーセンタイル基準を下回っており、筋肉量低下のリスクが示唆されます。",
            "interp_ratio": "内臓/皮下脂肪比高値 (≥ 1.0): 内臓脂肪の相対的蓄積を示唆し、代謝異常リスクと関連します。",
            "no_abnormal": "すべての評価指標は一般的な文献基準分布範囲内です。",
            "disclaimer_title": "3. 免責事項:",
            "disclaimer_text": "本システムで計算された画像解析指標は科学研究および臨床参考用であり、直接的な臨床診断結論をなすものではありません。",
            "ref_title": "4. 臨床基準および参考文献:",
            "pdf_ref_headers": ["指標名", "臨床的意義・特徴", "基準閾値", "引用文献"],
            "pdf_ref_data": [
                [
                    "<b>[1] 海綿骨平均密度</b><br/><font size='7' color='#4B5563'>(Cancellous Bone Atten.)</font>",
                    "骨粗鬆症および骨折リスクの機会的スクリーニング指標。",
                    "<b>L1 ≤ 110 HU</b><br/><font size='7' color='#4B5563'>(L3統一基準なし)</font>",
                    "Praveen AD, 2024;<br/>Pickhardt PJ, 2013"
                ],
                [
                    "<b>[2] 皮下脂肪指数</b><br/><font size='7' color='#4B5563'>(SATI)</font>",
                    "皮下脂肪蓄積の標準化指標。年齢やBMIの影響を受けます。",
                    "健康成人中央値:<br/>男性~38.8, 女性~55.8",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[3] 内臓脂肪指数</b><br/><font size='7' color='#4B5563'>(VATI)</font>",
                    "内臓脂肪蓄積の標準化指標。年齢とともに増加傾向。",
                    "健康成人中央値:<br/>男性~39.1, 女性~25.6",
                    "Kong M, et al.<br/>Front Nutr. 2022"
                ],
                [
                    "<b>[4] 骨格筋指数</b><br/><font size='7' color='#4B5563'>(SMI)</font>",
                    "骨格筋量の標準化指標。サルコペニア研究の基軸。",
                    "第5パーセンタイル:<br/><b>男&lt;40.2, 女&lt;31.6</b>",
                    "Kong M, et al. 2022;<br/>Prado CM, et al. 2008"
                ],
                [
                    "<b>[5] 内臓/皮下脂肪比</b><br/><font size='7' color='#4B5563'>(VAT/SAT Ratio)</font>",
                    "内臓脂肪の相対的蓄積を反映。代謝リスクと関連。",
                    "統一基準なし<br/><font size='7' color='#4B5563'>(対象群依存)</font>",
                    "Emamat H, et al.<br/>BMC Public Health. 2024"
                ],
                [
                    "<b>[6] 骨格筋密度</b><br/><font size='7' color='#4B5563'>(Muscle HU)</font>",
                    "筋脂肪化（Myosteatosis）および異所性脂肪浸潤を評価。",
                    "BMI&lt;25: <b>&lt; 41 HU</b><br/>BMI≥25: <b>&lt; 33 HU</b>",
                    "Garcia-Diez AI, 2024;<br/>Martin L, et al. 2013"
                ]
            ]
        }
    }
}

# ==============================
# 页面配置与基础设置
# ==============================
st.set_page_config(page_title="L3 AI Segmentation App", layout="wide")

# 侧边栏：语言切换
lang_map = {"中文": "zh", "English": "en", "日本語": "ja"}
lang_choice = st.sidebar.selectbox("🌐 语言 / Language", ["中文", "English", "日本語"])
lang = lang_map[lang_choice]
t = LANG[lang] 

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================
# 侧边栏：静态硬件状态信息
# ==============================
st.sidebar.markdown("---")
st.sidebar.header(t["hw_monitor"])
device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.write(f"**Engine**: `{device_type}`")

if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.write(f"**Hardware**: `{gpu_name}`")
    except Exception:
        pass

# ==============================
# 侧边栏：PACS 配置模块
# ==============================
st.sidebar.markdown("---")
st.sidebar.header(t["pacs_title"])
if not HAS_PRO_LIBS:
    st.sidebar.error("缺少 PACS 与 PDF 组件，请在终端执行:\npip install pydicom pynetdicom reportlab PyMuPDF")
else:
    pacs_ip = st.sidebar.text_input("目标 IP 地址", value="127.0.0.1")
    pacs_port = st.sidebar.text_input("目标端口", value="104")
    pacs_remote_aet = st.sidebar.text_input("目标 AET (Remote)", value="PACS")
    pacs_local_aet = st.sidebar.text_input("本地 AET (Local)", value="L3_Seg")
    
    if st.sidebar.button("📡 测试连接 (C-ECHO)", width='stretch'):
        try:
            ae = AE(ae_title=pacs_local_aet.encode('utf-8'))
            ae.requested_contexts = VerificationPresentationContexts
            assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=pacs_remote_aet.encode('utf-8'))
            if assoc.is_established:
                status = assoc.send_c_echo()
                assoc.release()
                if status:
                    st.sidebar.success("C-ECHO 成功！服务器通讯正常。")
                else:
                    st.sidebar.error("C-ECHO 失败！")
            else:
                st.sidebar.error("无法建立关联，请检查网络配置。")
        except Exception as e:
            st.sidebar.error(f"连接异常: {e}")

# ==============================
# 侧边栏：【智能动态寻址】权重路径配置
# ==============================
st.sidebar.markdown("---")
st.sidebar.header(t["sidebar_title"])

if getattr(sys, 'frozen', False):
    APP_ROOT = os.path.dirname(sys.executable)
else:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

default_ts_path = os.path.join(APP_ROOT, "Weights", "Dataset297_TotalSegmentator_total_3mm_1559subj")
default_nn_path = os.path.join(APP_ROOT, "Weights", "nnUNetTrainer__nnUNetPlans__2d", "fold_all")

TOTALSEG_WEIGHTS = st.sidebar.text_input(t["ts_weight"], value=default_ts_path)
NNUNET_WEIGHTS = st.sidebar.text_input(t["nnunet_weight"], value=default_nn_path)

# ==============================
# 核心功能模块 (分析与渲染)
# ==============================

def extract_dicom_to_nifti(zip_file, output_nii_path):
    saved_dcm_path = None
    workspace_dir = os.path.dirname(output_nii_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        dcm_dir = tmpdir
        first_dcm_src = None
        for root, dirs, files in os.walk(tmpdir):
            if any(f.lower().endswith(('.dcm', '.ima')) for f in files):
                dcm_dir = root
                first_dcm_src = os.path.join(root, [f for f in files if f.lower().endswith(('.dcm', '.ima'))][0])
                break
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir)
        if not dicom_names:
            raise ValueError("ZIP 压缩包中未找到有效的 DICOM 序列。")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, output_nii_path)
        
        if first_dcm_src:
            saved_dcm_path = os.path.join(workspace_dir, "reference_sample.dcm")
            shutil.copy2(first_dcm_src, saved_dcm_path)
    
    return saved_dcm_path

def extract_l3_slice(input_3d_nii, output_2d_nii, temp_dir):
    os.environ["TOTALSEG_WEIGHTS_PATH"] = os.path.dirname(TOTALSEG_WEIGHTS)
    os.environ["nnUNet_results"] = os.path.dirname(TOTALSEG_WEIGHTS)
    os.environ["TOTALSEGMENTATOR_DOWNLOAD_MODE"] = "off"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        from totalsegmentator.python_api import totalsegmentator
        totalsegmentator(
            input_3d_nii, 
            temp_dir, 
            task="total", 
            fast=True, 
            roi_subset=["vertebrae_L3"], 
            robust_crop=True, 
            quiet=True
        )
    except Exception as e:
        raise RuntimeError(f"TotalSegmentator 引擎调用失败: {str(e)}")
    
    l3_file = os.path.join(temp_dir, "vertebrae_L3.nii.gz")
    if not os.path.exists(l3_file):
        raise FileNotFoundError("未能成功生成 L3 掩码。")
    
    img_orig = nib.load(input_3d_nii)
    img_l3 = nib.load(l3_file)
    img_l3_resampled = resample_to_img(img_l3, img_orig, interpolation="nearest")
    
    coords = np.where(img_l3_resampled.get_fdata() > 0)
    if len(coords[2]) == 0:
        raise ValueError("L3 掩码为空，可能扫描范围未包含 L3 层面。")
    
    z_center = int(np.median(coords[2]))
    slice_data = img_orig.get_fdata()[:, :, z_center]
    
    slice_data_3d = np.expand_dims(slice_data, axis=-1) 
    sliced_img = nib.Nifti1Image(slice_data_3d, img_orig.affine)
    nib.save(sliced_img, output_2d_nii)
    return z_center

def run_nnunet_2d(input_file, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False, device=device, verbose=False, allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        os.path.dirname(NNUNET_WEIGHTS), use_folds=("all",), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files([[input_file]], output_dir, save_probabilities=False, overwrite=True)

def calculate_and_visualize(orig_nii_path, mask_nii_path, label_names, height_cm, sex, age):
    img = nib.load(orig_nii_path)
    msk = nib.load(mask_nii_path)
    img_ras = nib.as_closest_canonical(img)
    msk_ras = nib.as_closest_canonical(msk)
    data = np.squeeze(img_ras.get_fdata())
    mask_data = np.squeeze(np.rint(msk_ras.get_fdata()).astype(np.int32))
    spacing = img_ras.header.get_zooms()
    
    pixel_area_cm2 = (spacing[0] * spacing[1]) / 100.0
    height_m2 = (height_cm / 100.0) ** 2 
    
    results = []
    sat_area = vat_area = 0.0
    
    for label_val, label_name in label_names.items():
        voxel_mask = (mask_data == label_val)
        voxel_count = np.sum(voxel_mask)
        area_cm2 = voxel_count * pixel_area_cm2
        if label_val == 2: sat_area = area_cm2
        if label_val == 3: vat_area = area_cm2
        
        mean_hu = np.mean(data[voxel_mask]) if voxel_count > 0 else np.nan
        index_val = area_cm2 / height_m2 if (label_val in [2, 3, 4] and height_m2 > 0) else np.nan
        
        ref_text = eval_status = t["status_na"]
        if label_val == 1:
            # 椎体松质骨：明确标注为 L1 筛查参考线，L3 暂无统一截断值，不作直接判定
            ref_text = "L1 ≤ 110 HU [1]"
            eval_status = t["status_na"]
        elif label_val == 2:
            # SATI：北方中国健康人群中位数参考分布
            ref_text = "~ 55.8 [2]" if sex == "Female" else "~ 38.8 [2]"
            eval_status = t["status_na"]
        elif label_val == 3:
            # VATI：北方中国健康人群中位数参考分布
            ref_text = "~ 25.6 [3]" if sex == "Female" else "~ 39.1 [3]"
            eval_status = t["status_na"]
        elif label_val == 4:
            # SMI：北方中国健康成年人第 5 百分位低肌肉量标准（男 < 40.2，女 < 31.6）
            cutoff_smi = 31.6 if sex == "Female" else 40.2
            ref_text = f"< {cutoff_smi} [4]"
            if not np.isnan(index_val):
                eval_status = t["status_low_ref"] if index_val < cutoff_smi else t["status_normal"]
            
            # 第一行：骨骼肌总面积及 SMI 指标
            results.append({
                t["col_anat"]: label_name, 
                t["col_area"]: f"{area_cm2:.2f}",
                t["col_hu"]: "-",
                t["col_index"]: f"{index_val:.2f}" if not np.isnan(index_val) else "-",
                t["col_ref"]: ref_text,
                t["col_status"]: eval_status
            })
            
            # 第二行：骨骼肌平均密度 (Muscle HU)，提示结合 BMI 评估
            results.append({
                t["col_anat"]: t["col_muscle_hu"], 
                t["col_area"]: "-",
                t["col_hu"]: f"{mean_hu:.2f}" if not np.isnan(mean_hu) else "-",
                t["col_index"]: "-",
                t["col_ref"]: t["ref_muscle_hu"],
                t["col_status"]: t["status_check_bmi"]
            })
            continue 
            
        results.append({
            t["col_anat"]: label_name, 
            t["col_area"]: f"{area_cm2:.2f}",
            t["col_hu"]: f"{mean_hu:.2f}" if not np.isnan(mean_hu) else "-",
            t["col_index"]: f"{index_val:.2f}" if not np.isnan(index_val) else "-",
            t["col_ref"]: ref_text,
            t["col_status"]: eval_status
        })
    
    # VAT/SAT 比值：取消强制 high_risk 判定，客观呈现数值与文献引用
    vat_sat_ratio = (vat_area / sat_area) if sat_area > 0 else np.nan
    results.append({
        t["col_anat"]: "VAT / SAT Ratio", 
        t["col_area"]: "-", t["col_hu"]: "-",
        t["col_index"]: f"{vat_sat_ratio:.3f}" if not np.isnan(vat_sat_ratio) else "-",
        t["col_ref"]: t["ref_vatsat"], 
        t["col_status"]: t["status_na"]
    })
        
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
    s_img = np.flipud(np.fliplr(data.T))
    s_msk = np.flipud(np.fliplr(mask_data.T))
    aspect = spacing[1] / spacing[0]
    
    ax1.imshow(s_img, cmap='gray', vmin=-150, vmax=250, origin='upper', aspect=aspect)
    ax1.set_title(t["orig_img"], color='white')
    ax1.axis('off')

    ax2.imshow(s_img, cmap='gray', vmin=-150, vmax=250, origin='upper', aspect=aspect)
    ax2.set_title(t["seg_img"], color='white')
    ax2.axis('off')
    
    colors = ['red', 'lime', 'blue', 'yellow']
    for idx, l_val in enumerate(np.unique(s_msk)[1:]):
        ax2.contour(s_msk == l_val, levels=[0.5], colors=colors[idx % 4], linewidths=1.2)
    fig.tight_layout()
    return fig, df_results

# ==============================
# 一页 A4 三线表 + 动态引文解析 PDF 引擎
# ==============================
def generate_pdf_report(patient_info, fig, df_metrics, t_lang):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=25, bottomMargin=25)
    elements = []
    
    style_title = ParagraphStyle(name='ST-Title', fontName='STSong-Light', fontSize=18, alignment=1, spaceAfter=10)
    style_normal = ParagraphStyle(name='ST-Normal', fontName='STSong-Light', fontSize=9.5, spaceAfter=3, leading=13)
    style_bold = ParagraphStyle(name='ST-Bold', fontName='STSong-Light', fontSize=11, textColor=colors.black, spaceAfter=4, leading=14) 
    style_small = ParagraphStyle(name='ST-Small', fontName='STSong-Light', fontSize=8, spaceAfter=2, leading=11)
    style_ref_cell = ParagraphStyle(name='ST-RefCell', fontName='STSong-Light', fontSize=7.5, leading=10)
    style_ref_header = ParagraphStyle(name='ST-RefHeader', fontName='STSong-Light', fontSize=8.5, leading=11, alignment=1)

    elements.append(Paragraph(f'<b>{t_lang["pdf"]["report_title"]}</b>', style_title))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_str = t_lang["pdf"]["report_time"].format(time=current_time)
    p_info_str = t_lang["pdf"]["patient_info"].format(**patient_info)
    info_html = f"{p_info_str} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {time_str}"
    elements.append(Paragraph(info_html, style_normal))

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='jpg', dpi=150, bbox_inches='tight', facecolor='black')
    img_buf.seek(0)
    img = RLImage(img_buf)
    
    avail_width = A4[0] - 60  
    aspect = img.imageHeight / float(img.imageWidth)
    img.drawWidth = avail_width
    img.drawHeight = avail_width * aspect
    
    # 控制图片高度，确保严格维持 A4 单页布局
    if img.drawHeight > 150:
        img.drawHeight = 150
        img.drawWidth = 150 / aspect
        
    elements.append(img)
    elements.append(Spacer(1, 4))

    elements.append(Paragraph(f'<b>{t_lang["pdf"]["table_title"]}</b>', style_bold))
    headers = [t_lang["col_anat"], t_lang["col_area"], t_lang["col_hu"], t_lang["col_index"], t_lang["col_ref"], t_lang["col_status"]]
    data = [headers]
    
    clinical_notes = []
    for index, row in df_metrics.iterrows():
        row_data = [str(item) for item in row.values]
        data.append(row_data)
        anat_name = row_data[0]
        hu_str = row_data[2]
        idx_str = row_data[3]
        
        # 智能临床参考提示提取
        if ("Cancellous" in anat_name or "松质骨" in anat_name or "海綿骨" in anat_name) and hu_str != "-":
            try:
                if float(hu_str) < 110.0:
                    clinical_notes.append("• " + t_lang["pdf"]["interp_bone"])
            except ValueError:
                pass
        elif ("Muscle" in anat_name or "骨骼肌" in anat_name or "骨格筋" in anat_name) and "Density" not in anat_name and "密度" not in anat_name:
            if row_data[5] == t_lang["status_low_ref"]:
                clinical_notes.append("• " + t_lang["pdf"]["interp_muscle"])
        elif "VAT / SAT" in anat_name and idx_str != "-":
            try:
                if float(idx_str) >= 1.0:
                    clinical_notes.append("• " + t_lang["pdf"]["interp_ratio"])
            except ValueError:
                pass

    table = Table(data, colWidths=[105, 80, 80, 80, 110, 80])
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'STSong-Light'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('TOPPADDING', (0,0), (-1,0), 3),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2),
        ('TOPPADDING', (0,1), (-1,-1), 2),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 4))

    elements.append(Paragraph(f'<b>{t_lang["pdf"]["interp_title"]}</b>', style_bold))
    if not clinical_notes:
        elements.append(Paragraph(t_lang["pdf"]["no_abnormal"], style_normal))
    else:
        for note in clinical_notes:
            elements.append(Paragraph(note, style_normal))
    elements.append(Spacer(1, 3))

    elements.append(Paragraph(f'<b>{t_lang["pdf"]["disclaimer_title"]}</b>', style_bold))
    elements.append(Paragraph(t_lang["pdf"]["disclaimer_text"], style_small))
    elements.append(Spacer(1, 3))

    elements.append(Paragraph(f'<b>{t_lang["pdf"]["ref_title"]}</b>', style_bold))
    
    ref_headers = t_lang["pdf"]["pdf_ref_headers"]
    ref_data_raw = t_lang["pdf"]["pdf_ref_data"]

    ref_table_data = [[Paragraph(f"<b>{h}</b>", style_ref_header) for h in ref_headers]]
    for row in ref_data_raw:
        ref_table_data.append([Paragraph(cell, style_ref_cell) for cell in row])

    ref_table = Table(ref_table_data, colWidths=[85, 215, 115, 120])
    ref_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'STSong-Light'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'CENTER'), 
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(ref_table)
        
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ==============================
# DICOM 封装生成器 (SC & PDF 转 SC)
# ==============================
def create_pacs_dicoms(ref_dcm_path, fig, pdf_bytes):
    base_ds = pydicom.dcmread(ref_dcm_path)
    
    def clean_dicom_tag(dcm_obj, tag_name):
        if tag_name in dcm_obj:
            val = dcm_obj.data_element(tag_name).value
            if isinstance(val, str):
                dcm_obj.data_element(tag_name).value = val.replace('\x00', '').strip()
                
    clean_dicom_tag(base_ds, 'PatientName')
    clean_dicom_tag(base_ds, 'PatientID')
    clean_dicom_tag(base_ds, 'StudyInstanceUID')
    clean_dicom_tag(base_ds, 'SeriesDescription')
    
    dicoms_to_send = []
    
    def make_sc_rgb_dicom(img_arr_rgb, series_uid, series_desc, series_num, instance_num):
        ds = base_ds.copy()
        
        if not hasattr(ds, 'file_meta'):
            ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        for tag_name in ['WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope', 'PixelPaddingValue']:
            try:
                if tag_name in ds:
                    del ds[tag_name]
            except Exception:
                pass
                
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = series_desc
        ds.Modality = "OT"
        ds.ConversionType = "WSD"
        ds.SeriesNumber = series_num
        ds.InstanceNumber = instance_num
        
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0
        ds.NumberOfFrames = 1
        ds.Rows, ds.Columns = img_arr_rgb.shape[:2]
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = img_arr_rgb.tobytes()
        
        return ds

    common_series_uid = generate_uid()
    common_series_num = int(base_ds.SeriesNumber) + 1000 if 'SeriesNumber' in base_ds else 1000
    common_series_desc = "L3 AI Segmentation & Report"

    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    
    dicoms_to_send.append(
        make_sc_rgb_dicom(img_array, common_series_uid, common_series_desc, common_series_num, 1)
    )
    
    if HAS_PRO_LIBS and 'fitz' in sys.modules:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=150)
                page_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if pix.n == 4:
                    page_arr = page_arr[:, :, :3]
                elif pix.n == 1:
                    page_arr = np.stack((page_arr.squeeze(),)*3, axis=-1)
                    
                dicoms_to_send.append(
                    make_sc_rgb_dicom(page_arr, common_series_uid, common_series_desc, common_series_num, page_idx + 2)
                )
            doc.close()
        except Exception as e:
            logging.error(f"PDF 栅格化失败: {e}")
            pass
            
    return dicoms_to_send

# ==============================
# Streamlit 主界面交互逻辑
# ==============================
st.write(t["upload_title"])

uploaded_file = st.file_uploader(t["upload_hint"], type=["zip", "gz", "nii"])

col_h, col_s, col_a = st.columns(3)
with col_h:
    patient_height_cm = st.number_input(t["height_input"], min_value=50.0, max_value=300.0, value=170.0, step=1.0)
with col_s:
    patient_sex = st.selectbox(t["sex_input"], ["Male", "Female"], format_func=lambda x: "男 / Male" if x=="Male" else "女 / Female")
with col_a:
    patient_age = st.number_input(t["age_input"], min_value=0, max_value=120, value=50, step=1)

LABEL_DICT = t["labels"]

if uploaded_file is not None:
    if st.button(t["btn_start"], type="primary", width='stretch'):
        with tempfile.TemporaryDirectory() as workspace:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info(t["status_parse"])
            input_3d_nii = os.path.join(workspace, "input_3d.nii.gz")
            ref_dcm_path = None
            patient_name_dcm = "Unknown"
            
            if uploaded_file.name.endswith(".zip"):
                with open(os.path.join(workspace, "upload.zip"), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                ref_dcm_path = extract_dicom_to_nifti(os.path.join(workspace, "upload.zip"), input_3d_nii)
                if HAS_PRO_LIBS and ref_dcm_path:
                    ds_tmp = pydicom.dcmread(ref_dcm_path, stop_before_pixels=True)
                    p_name_raw = str(ds_tmp.PatientName) if 'PatientName' in ds_tmp else "Unknown"
                    patient_name_dcm = p_name_raw.replace('\x00', '').strip() or "Unknown"
            else:
                with open(input_3d_nii, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                fname = uploaded_file.name
                if fname.endswith(".nii.gz"): fname = fname[:-7]
                elif fname.endswith(".nii"): fname = fname[:-4]
                patient_name_dcm = fname.replace('\x00', '').strip() or "Unknown"
                    
            progress_bar.progress(15)
            
            output_2d_nii = os.path.join(workspace, "l3_slice_0000.nii.gz")
            ts_tmp_dir = os.path.join(workspace, "ts_output")
            with st.spinner(t["status_ts"]):
                try:
                    z_idx = extract_l3_slice(input_3d_nii, output_2d_nii, ts_tmp_dir)
                    st.success(f"{t['success_l3']} {z_idx}")
                except Exception as e:
                    st.error(f"{t['err_l3']} {str(e)}")
                    st.stop()
            progress_bar.progress(55)
            
            nnunet_out_dir = os.path.join(workspace, "nnunet_output")
            os.makedirs(nnunet_out_dir, exist_ok=True)
            with st.spinner(t["status_nnunet"]):
                try:
                    run_nnunet_2d(output_2d_nii, nnunet_out_dir)
                except Exception as e:
                    st.error(f"{t['err_nnunet']} {str(e)}")
                    st.stop()
            progress_bar.progress(85)
            
            status_text.info(t["status_calc"])
            pred_mask = os.path.join(nnunet_out_dir, "l3_slice.nii.gz")
            fig, df_metrics = calculate_and_visualize(output_2d_nii, pred_mask, LABEL_DICT, patient_height_cm, patient_sex, patient_age)
            
            pdf_bytes = None
            if HAS_PRO_LIBS:
                sex_str = {"zh": "男" if patient_sex=="Male" else "女", 
                           "en": patient_sex, 
                           "ja": "男" if patient_sex=="Male" else "女"}[lang]
                           
                p_info = {'name': patient_name_dcm, 'sex': sex_str, 'age': patient_age, 'height': patient_height_cm}
                
                pdf_bytes = generate_pdf_report(p_info, fig, df_metrics, t)
                
                if ref_dcm_path:
                    st.session_state['dicoms_to_send'] = create_pacs_dicoms(ref_dcm_path, fig, pdf_bytes)
            
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(t["csv_filename"], df_metrics.to_csv(index=False).encode("utf-8-sig"))
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='jpg', bbox_inches='tight', dpi=150, facecolor='black')
                zipf.writestr("L3_visualization.jpg", img_buf.getvalue())
                zipf.write(output_2d_nii, arcname="L3_slice_original.nii.gz")
                zipf.write(pred_mask, arcname="L3_slice_mask.nii.gz")
                if pdf_bytes:
                    zipf.writestr("L3_Clinical_Report.pdf", pdf_bytes)
            
            progress_bar.progress(100)
            status_text.empty() 
            
            st.write("---")
            st.write(t["panel_title"])
            st.warning(t["disclaimer"])
            
            col1, col2 = st.columns([1.1, 1.6]) 
            with col1:
                st.pyplot(fig)
            with col2:
                st.write(t["table_title"])
                st.dataframe(df_metrics, use_container_width=True)
                
                st.download_button(
                    label=t["btn_zip"] + (" (含 PDF 报告)" if pdf_bytes else ""),
                    data=zip_buf.getvalue(),
                    file_name="L3_Analysis_Package.zip",
                    mime="application/zip",
                    width='stretch'
                )
            
            st.write("---")
            with st.expander(t["citations_title"], expanded=True):
                st.markdown(t["citations_text"])

# ==============================
# PACS 上传触发逻辑
# ==============================
if 'dicoms_to_send' in st.session_state and st.session_state['dicoms_to_send']:
    st.write("---")
    st.subheader("📤 PACS 快速归档")
    if st.button("将生成的图像与 PDF 报告一键发送至 PACS", type="primary", width='stretch'):
        try:
            ae = AE(ae_title=pacs_local_aet.encode('utf-8'))
            
            contexts_dict = {}
            for d in st.session_state['dicoms_to_send']:
                sop = d.SOPClassUID
                tx = d.file_meta.TransferSyntaxUID if hasattr(d, 'file_meta') else pydicom.uid.ImplicitVRLittleEndian
                if sop not in contexts_dict:
                    contexts_dict[sop] = set()
                contexts_dict[sop].add(tx)
                contexts_dict[sop].add(pydicom.uid.ExplicitVRLittleEndian)
                contexts_dict[sop].add(pydicom.uid.ImplicitVRLittleEndian)

            custom_contexts = []
            for sop_class, tx_list in contexts_dict.items():
                custom_contexts.append(build_context(sop_class, transfer_syntax=list(tx_list)))
            
            ae.requested_contexts = custom_contexts
            assoc = ae.associate(pacs_ip, int(pacs_port), ae_title=pacs_remote_aet.encode('utf-8'))
            
            if assoc.is_established:
                sent_count = 0
                total_dcm = len(st.session_state['dicoms_to_send'])
                progress_pacs = st.progress(0)
                
                for i, ds in enumerate(st.session_state['dicoms_to_send']):
                    status = assoc.send_c_store(ds)
                    if status and status.Status in [0x0000, 0xB000, 0xB006, 0xB007]:
                        sent_count += 1
                    progress_pacs.progress((i + 1) / total_dcm)
                    
                assoc.release()
                
                if sent_count == total_dcm:
                    st.success(f"🎉 成功！已将 {sent_count} 个结果文件（图像+报告）归档至同一检查 (Study) 下的同一序列中。")
                else:
                    st.warning(f"⚠️ 发送完毕，但仅有 {sent_count}/{total_dcm} 个文件被 PACS 接受。")
            else:
                st.error("❌ 无法建立 PACS 连接，请检查侧边栏的网络参数。")
        except Exception as e:
            st.error(f"❌ 传输发生系统异常: {e}")