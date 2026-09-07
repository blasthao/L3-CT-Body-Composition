import math
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ==========================================
# 0. 多语言配置字典
# ==========================================
LANG = {
    "zh": {
        "title": "基于CT的L3组织分割预测糖尿病风险的模型",
        "subtitle": "基于多中心回顾性队列及多因素 Logistic 回归构建的临床辅助诊断工具。",
        "sidebar_title": "📋 自动提取的体成分数据",
        "sidebar_caption": "提示：在完整版中，这些数据将由 CT 分割流水线自动填入。",
        "lbl_bmi": "体质指数 (BMI, kg/m²)",
        "lbl_age": "患者年龄 (Age)",
        "lbl_muscle": "骨骼肌密度 (Muscle HU)",
        "lbl_smi": "骨骼肌指数 (SMI, cm²/m²)",
        "lbl_vatsat": "内脏/皮下脂肪比 (VAT/SAT Ratio)",
        "lbl_bone": "松质骨密度 (Bone HU)",
        "box_prob": "预测患病概率 (Prediction)",
        "box_eval": "临床评估结论",
        "disclaimer": "**免责声明**：本模型基于回顾性数据构建，计算结果仅供科研与临床医生参考，**不构成明确的医疗诊断依据**。请结合患者真实体征及生化检验指标进行最终评估。",
        "alert_high": "⚠️ **高风险提示**：该患者预测概率超过截断值 ({threshold}%)。影像学体成分表现出一定的代谢异常倾向，提示并发糖尿病的风险较高。建议结合临床实际及空腹血糖等生化指标进一步评估。",
        "alert_low": "✅ **低风险提示**：该患者预测概率低于截断值 ({threshold}%)。当前 CT 体成分结构未见显著异常，提示并发糖尿病的风险相对较低。",
        "ref_title": "📌 核心体成分指标参考与提示",
        "info_muscle": "💪 **骨骼肌密度 (质量)**：当前处于文献常规参考区间。文献 BMI 分层参考：BMI < 25 时应 ≥ 41 HU，BMI ≥ 25 时应 ≥ 33 HU（主要源于肿瘤人群研究）。",
        "warn_muscle": "💪 **骨骼肌密度提示 (Myosteatosis)**：结合患者 BMI，当前骨骼肌密度低于文献分层参考值（BMI < 25 时 < 41 HU，BMI ≥ 25 时 < 33 HU）。提示可能存在肌纤维内异位脂质沉积，与胰岛素抵抗及不良代谢紧密相关。",
        "info_bone": "🦴 **松质骨密度**：当前 ≥ 110 HU。高于经典 L1 机会性筛查参考线（注：尚无公认统一的 L3 临床截断值）。",
        "warn_bone": "🦴 **松质骨密度提示**：当前 < 110 HU。低于经典 L1 机会性骨质疏松筛查参考线（110 HU），提示关注骨质流失风险（注：该阈值源自 L1 椎体，不宜直接外推至 L3）。",
        "info_vatsat": "🫀 **内脏/皮下脂肪比**：当前 < 1.0。脂肪分布更倾向于皮下脂肪蓄积（注：尚无适用于不同人群的统一临床截断值）。",
        "warn_vatsat": "🫀 **内脏/皮下脂肪比提示**：当前 ≥ 1.0。提示腹部脂肪分布显著偏向内脏脂肪，与不良心血管及糖脂代谢表型相关（注：尚无统一临床截断值）。",
        "info_smi": "💪 **骨骼肌指数 (SMI)**：北方中国健康成年人第 5 百分位低值参考为：男性 < 40.2 cm²/m²，女性 < 31.6 cm²/m²（主要用于科研与低肌量评估，不作为通用肌少症诊断标准）。",
        "plot1_title": "🔍 个体化预测动态列线图 (SHAP Waterfall)",
        "plot1_caption": "实时展示当前患者各项生理指标是如何将基础风险推高(红)或拉低(蓝)的。",
        "plot2_title": "📊 队列特征分布影响图 (SHAP Beeswarm)",
        "plot2_caption": "基于 300 例开发集队列生成。红点代表该特征数值较高，分布在右侧代表增加患病风险。",
        "feat_base": "基础风险 (Intercept)",
        "feat_muscle": "Muscle (HU)",
        "feat_smi": "SMI",
        "feat_vatsat": "VAT/SAT Ratio",
        "feat_age": "Age",
        "feat_bone": "Bone (HU)",
        "feat_total": "总计 (Logit)",
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
        """
    },
    "en": {
        "title": "CT-based L3 Tissue Segmentation Model for Diabetes Risk Prediction",
        "subtitle": "A clinical diagnostic support tool based on multi-center cohorts and multivariable logistic regression.",
        "sidebar_title": "📋 Extracted Body Composition Data",
        "sidebar_caption": "Note: In the full version, these metrics will be automatically populated.",
        "lbl_bmi": "Body Mass Index (BMI, kg/m²)",
        "lbl_age": "Patient Age",
        "lbl_muscle": "Skeletal Muscle Density (HU)",
        "lbl_smi": "Skeletal Muscle Index (SMI, cm²/m²)",
        "lbl_vatsat": "VAT/SAT Ratio",
        "lbl_bone": "Cancellous Bone Density (HU)",
        "box_prob": "Predicted Probability",
        "box_eval": "Clinical Evaluation",
        "disclaimer": "**Disclaimer**: This model is based on retrospective data. Results are for reference only, and **do not constitute a definitive medical diagnosis**. Please evaluate in conjunction with actual clinical signs and tests.",
        "alert_high": "⚠️ **High Risk Alert**: The predicted probability exceeds the threshold ({threshold}%). Imaging features show tendencies of metabolic abnormality, suggesting a higher risk of diabetes. Further clinical evaluation is recommended.",
        "alert_low": "✅ **Low Risk**: The predicted probability is below the threshold ({threshold}%). Current CT body composition suggests a relatively low risk of concurrent diabetes.",
        "ref_title": "📌 Core Body Composition Reference & Prompts",
        "info_muscle": "💪 **Muscle Density**: Within typical reference range. Literature BMI-stratified cutoffs: ≥ 41 HU if BMI < 25, ≥ 33 HU if BMI ≥ 25 (primarily from oncology cohorts).",
        "warn_muscle": "💪 **Myosteatosis Prompt**: Muscle density is below literature reference cutoffs (< 41 HU if BMI < 25, < 33 HU if BMI ≥ 25). Suggests potential ectopic intramyocellular lipid infiltration, associated with insulin resistance.",
        "info_bone": "🦴 **Cancellous Bone**: Currently ≥ 110 HU. Above classic L1 opportunistic screening benchmark (Note: No established universal L3 cutoff).",
        "warn_bone": "🦴 **Cancellous Bone Prompt**: Currently < 110 HU. Below classic L1 opportunistic osteoporosis screening line (110 HU), indicating risk of bone loss (Note: L1 threshold cannot be directly generalized to L3).",
        "info_vatsat": "🫀 **VAT/SAT Ratio**: Currently < 1.0. Indicates subcutaneous-predominant adipose distribution (Note: No unified clinical cutoff across diverse populations).",
        "warn_vatsat": "🫀 **Adiposity Prompt**: VAT/SAT ≥ 1.0. Indicates visceral-predominant adipose accumulation, linked to unfavorable cardiometabolic profiles (Note: No universal cutoff).",
        "info_smi": "💪 **Skeletal Muscle Index (SMI)**: 5th percentile reference for healthy northern Chinese adults: Male < 40.2 cm²/m², Female < 31.6 cm²/m² (For population research/low muscle mass evaluation; not a universal diagnostic criterion).",
        "plot1_title": "🔍 Individualized Prediction (SHAP Waterfall)",
        "plot1_caption": "Real-time visualization of how each metric pushes the base risk higher (red) or lower (blue).",
        "plot2_title": "📊 Cohort Feature Distribution (SHAP Beeswarm)",
        "plot2_caption": "Generated from the 300-patient development cohort. Red dots represent higher feature values.",
        "feat_base": "Base Risk (Intercept)",
        "feat_muscle": "Muscle (HU)",
        "feat_smi": "SMI",
        "feat_vatsat": "VAT/SAT Ratio",
        "feat_age": "Age",
        "feat_bone": "Bone (HU)",
        "feat_total": "Total (Logit)",
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
        """
    },
    "ja": {
        "title": "CTのL3組織セグメンテーションに基づく糖尿病リスク予測モデル",
        "subtitle": "多施設後方視的コホートおよび多変量ロジスティック回帰に基づく臨床診断支援ツール。",
        "sidebar_title": "📋 抽出された体組成データ",
        "sidebar_caption": "ヒント：完全版では、これらのデータはAIによって自動的に入力されます。",
        "lbl_bmi": "ボディマス指数 (BMI, kg/m²)",
        "lbl_age": "患者の年齢 (Age)",
        "lbl_muscle": "骨格筋密度 (Muscle HU)",
        "lbl_smi": "骨格筋指数 (SMI, cm²/m²)",
        "lbl_vatsat": "内臓/皮下脂肪比 (VAT/SAT Ratio)",
        "lbl_bone": "海綿骨密度 (Bone HU)",
        "box_prob": "予測される罹患確率",
        "box_eval": "臨床評価の結論",
        "disclaimer": "**免責事項**：本モデルは後方視的データに基づいています。計算結果は科学研究および医師の参考用であり、**明確な医学的診断の根拠を構成するものではありません**。",
        "alert_high": "⚠️ **高リスク警告**：予測確率がカットオフ値 ({threshold}%) を超えています。画像上の体組成は代謝異常の傾向を示しており、糖尿病併発のリスクが高いことが示唆されます。",
        "alert_low": "✅ **低リスクのヒント**：予測確率がカットオフ値 ({threshold}%) を下回っており、糖尿病を併発するリスクは比較的低いことが示唆されます。",
        "ref_title": "📌 コア体組成指標の参考と提示",
        "info_muscle": "💪 **骨格筋密度 (Muscle HU)**: 文献基準範囲内です。文献的BMI層別基準：BMI<25で≥41 HU、BMI≥25で≥33 HU（主に悪性腫瘍コホートに基づく）。",
        "warn_muscle": "💪 **筋脂肪化 (Myosteatosis) 提示**: BMI層別基準を下回っています（BMI<25: <41 HU, BMI≥25: <33 HU）。筋肉内異所性脂肪沈着を示唆し、インスリン抵抗性や代謝異常と関連します。",
        "info_bone": "🦴 **海綿骨密度**: 現在 ≥110 HU。古典的L1スクリーニング基準以上です（注：L3統一基準は未確立）。",
        "warn_bone": "🦴 **海綿骨密度提示**: 現在 <110 HU。古典的L1骨粗鬆症スクリーニング基準（110 HU）を下回っており、骨量減少リスクに留意が必要です（注：L1基準を直接L3に適用不可）。",
        "info_vatsat": "🫀 **内臓/皮下脂肪比**: 現在 <1.0。皮下脂肪優位の蓄積を示唆します（注：統一基準なし）。",
        "warn_vatsat": "🫀 **内臓/皮下脂肪比提示**: 現在 ≥1.0。内臓脂肪優位の蓄積を示唆し、心血管・代謝リスクと関連します（注：統一基準なし）。",
        "info_smi": "💪 **骨格筋指数 (SMI)**: 中国北方健康成人の第5パーセンタイル基準値：男性 < 40.2 cm²/m²、女性 < 31.6 cm²/m²（研究および低筋肉量評価用であり、普遍的診断基準ではありません）。",
        "plot1_title": "🔍 個別化予測の説明 (SHAP ウォーターフォール図)",
        "plot1_caption": "各生理学的指標が基礎リスクをどのように押し上げ（赤）たり引き下げ（青）たりしているかをリアルタイムで視覚化します。",
        "plot2_title": "📊 コホート特徴分布 (SHAP ビースウォーム図)",
        "plot2_caption": "300例の開発コホートから生成されました。赤い点は高い数値を示します。",
        "feat_base": "基礎リスク",
        "feat_muscle": "Muscle (HU)",
        "feat_smi": "SMI",
        "feat_vatsat": "VAT/SAT Ratio",
        "feat_age": "Age",
        "feat_bone": "Bone (HU)",
        "feat_total": "合計 (Logit)",
        "citations_title": "📚 臨床基準と参考文献索引",
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
        """
    }
}

# ==========================================
# 1. 页面与基础配置
# ==========================================
st.set_page_config(page_title="AI 糖尿病筛查系统", layout="wide", page_icon="🩺")

st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; color: #1E3A8A; }
    .risk-high { color: #DC2626; font-size: 95px !important; font-weight: 900; line-height: 1.2 !important; margin: 0; text-shadow: 2px 4px 8px rgba(0,0,0,0.1); white-space: nowrap;}
    .risk-low { color: #16A34A; font-size: 95px !important; font-weight: 900; line-height: 1.2 !important; margin: 0; text-shadow: 2px 4px 8px rgba(0,0,0,0.1); white-space: nowrap;}
    .st-emotion-cache-1v0mbdj { border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    [data-testid="stSidebar"] .stSlider label { font-size: 20px !important; font-weight: 900 !important; color: #1E3A8A !important; }
    [data-testid="stSidebar"] .stSlider div[data-testid="stThumbValue"] { font-size: 18px !important; font-weight: bold !important; color: #DC2626 !important;}
    </style>
""", unsafe_allow_html=True)

lang_choice = st.sidebar.selectbox("🌐 语言 / Language", ["中文", "English", "日本語"])
lang_map = {"中文": "zh", "English": "en", "日本語": "ja"}
t = LANG[lang_map[lang_choice]]

# ==========================================
# 2. 多因素 Logistic 回归参数
# ==========================================
COEFS = {
    "Intercept": 0.552,
    "Age": 0.010,
    "Bone_HU": 0.001,
    "SMI": 0.051,
    "Muscle_HU": -0.126,
    "VAT_SAT_Ratio": 0.563
}
THRESHOLD = 0.563

# ==========================================
# 3. 后台加载队列数据并生成全局 SHAP (加入缓存机制确保极速流畅)
# ==========================================
@st.cache_data
def load_global_shap():
    try:
        try:
            df = pd.read_csv('shap.csv', encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv('shap.csv', encoding='gbk')
            
        df.columns = df.columns.str.strip()
        feature_cols = ['Age', 'Bone (HU)', 'SMI', 'Muscle (HU)', 'VAT/SAT Ratio']
        X = df[feature_cols]
        y = df['DM_Status']
        
        try:
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
        except ValueError:
            model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=2000)
            
        model.fit(X, y)
        explainer = shap.Explainer(model, X)
        shap_values_global = explainer(X)
        return shap_values_global
    except Exception as e:
        return None

global_shap_values = load_global_shap()

# ==========================================
# 4. 个体计算逻辑与即刻生成 SHAP Explanation 对象
# ==========================================
def calculate_risk_and_shap(muscle_hu, smi, vat_sat, age, bone_hu):
    logit = (COEFS["Intercept"] + 
             COEFS["Muscle_HU"] * muscle_hu + 
             COEFS["SMI"] * smi + 
             COEFS["VAT_SAT_Ratio"] * vat_sat + 
             COEFS["Age"] * age + 
             COEFS["Bone_HU"] * bone_hu)
    prob = 1 / (1 + math.exp(-logit))
    
    shap_vals = np.array([
        age * COEFS["Age"],
        bone_hu * COEFS["Bone_HU"],
        smi * COEFS["SMI"],
        muscle_hu * COEFS["Muscle_HU"],
        vat_sat * COEFS["VAT_SAT_Ratio"]
    ])
    
    feature_names = [t["feat_age"], t["feat_bone"], t["feat_smi"], t["feat_muscle"], t["feat_vatsat"]]
    feature_data = np.round(np.array([age, bone_hu, smi, muscle_hu, vat_sat]), 2)
    
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=COEFS["Intercept"],
        data=feature_data,
        feature_names=feature_names
    )
    
    return prob, logit, explanation

# ==========================================
# 5. 侧边栏交互
# ==========================================
st.sidebar.markdown(f"### {t['sidebar_title']}")
st.sidebar.caption(t['sidebar_caption'])

val_bmi = st.sidebar.slider(t["lbl_bmi"], min_value=15.0, max_value=40.0, value=24.0, step=0.1)
val_age = st.sidebar.slider(t["lbl_age"], min_value=20, max_value=90, value=65, step=1, format="%d")
val_muscle_hu = st.sidebar.slider(t["lbl_muscle"], min_value=10.0, max_value=60.0, value=35.0, step=0.1)
val_smi = st.sidebar.slider(t["lbl_smi"], min_value=20.0, max_value=80.0, value=45.0, step=0.1)
val_vat_sat = st.sidebar.slider(t["lbl_vatsat"], min_value=0.1, max_value=4.0, value=1.12, step=0.01)
val_bone_hu = st.sidebar.slider(t["lbl_bone"], min_value=50.0, max_value=250.0, value=93.0, step=1.0)

# ==========================================
# 6. UI 顶部大盘
# ==========================================
st.title(t["title"])
st.markdown(t["subtitle"])
st.divider()

prob, logit, shap_exp = calculate_risk_and_shap(val_muscle_hu, val_smi, val_vat_sat, val_age, val_bone_hu)

col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.markdown(f"<p class='big-font'>{t['box_prob']}</p>", unsafe_allow_html=True)
    if prob >= THRESHOLD:
        st.markdown(f"<p class='risk-high'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='risk-low'>{prob*100:.1f}%</p>", unsafe_allow_html=True)
    st.warning(t["disclaimer"], icon="⚠️")

with col2:
    st.markdown(f"<p class='big-font'>{t['box_eval']}</p>", unsafe_allow_html=True)
    if prob >= THRESHOLD:
        st.error(t["alert_high"].format(threshold=round(THRESHOLD*100, 1)))
    else:
        st.success(t["alert_low"].format(threshold=round(THRESHOLD*100, 1)))
        
    st.markdown(f"##### {t['ref_title']}")
    muscle_threshold = 41.0 if val_bmi < 25.0 else 33.0
    if val_muscle_hu < muscle_threshold:
        st.warning(t["warn_muscle"])
    else:
        st.info(t["info_muscle"])
    if val_bone_hu < 110.0:
        st.warning(t["warn_bone"])
    else:
        st.info(t["info_bone"])
    if val_vat_sat >= 1.0:
        st.warning(t["warn_vatsat"])
    else:
        st.info(t["info_vatsat"])
    st.info(t["info_smi"])

st.divider()

# ==========================================
# 7. 交互式可视化分析 (双 SHAP 顶级图表)
# ==========================================
col_plot1, col_plot2 = st.columns([1.1, 0.9])

with col_plot1:
    st.markdown(f"#### {t['plot1_title']}")
    st.caption(t["plot1_caption"])
    
    # 渲染个体预测瀑布图
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    shap.plots.waterfall(shap_exp, show=False, max_display=10)
    plt.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close()

with col_plot2:
    st.markdown(f"#### {t['plot2_title']}")
    st.caption(t["plot2_caption"])
    
    # 渲染全局分布蜂窝图
    if global_shap_values is not None:
        fig_bee, ax_bee = plt.subplots(figsize=(6, 4), facecolor='none')
        shap.plots.beeswarm(global_shap_values, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee, transparent=True)
        plt.close()
    else:
        st.error("无法加载 shap.csv 生成队列蜂窝图，请确保数据文件在同级目录。")

# ==========================================
# 8. 预测指标的临床解释与文献背书
# ==========================================
st.write("---")
with st.expander(t["citations_title"], expanded=True):
    st.markdown(t["citations_text"])