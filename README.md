# End-to-End Opportunistic CT Body Composition Profiling & Interpretable Diabetes Risk Prediction Suites

An end-to-end, open-source software suite for opportunistic body composition analysis and metabolic risk estimation using routine non-contrast abdominal computed tomography (NCCT) scans. The pipeline integrates automated vertebral localization, 4-class multi-tissue pixel-level segmentation, standardized metric quantification, DICOM 3.0-compliant PACS workflow integration, and an explainable AI (XAI) dashboard for downstream type 2 diabetes risk stratification powered by SHAP.

---

## 🌟 Key Features

The system consists of two interconnected, zero-code lightweight web applications:

### 1. App 1: Automated L3 Segmentation & Clinical PACS Workflow (`L3_app.py`)

* **Flexible Input Ingestion**: Supports direct upload of raw abdominal CT series in either `DICOM ZIP archive` or `3D NIfTI (.nii.gz)` formats.


* **Automated Anatomical Localization**: Leverages official TotalSegmentator weights (`Dataset297_TotalSegmentator_total_3mm_1559subj`) to identify the L3 vertebral body in 3D space and extract the mid-vertebral axial slice, eliminating manual slice selection.


* **2D nnU-Net Multi-Tissue Segmentation**: Performs simultaneous pixel-level semantic segmentation of four key anatomical compartments on the L3 slice: **Cancellous Bone**, **Subcutaneous Adipose Tissue (SAT)**, **Visceral Adipose Tissue (VAT)**, and **Skeletal Muscle**.


* **Standardized Quantitative Profiling**:
* **Cross-Sectional Areas & Derived Indices**: Calculates physical cross-sectional areas ($\text{cm}^2$) and normalizes them by squared height ($\text{Height}^2$) to yield the Skeletal Muscle Index ($\text{SMI}$), Subcutaneous Adipose Tissue Index ($\text{SATI}$), Visceral Adipose Tissue Index ($\text{VATI}$), and the Visceral-to-Subcutaneous Adipose Tissue Ratio ($\text{VAT/SAT Ratio}$).


* **Radiodensity Assessment**: Quantifies mean attenuation in Hounsfield Units (HU) for cancellous bone and skeletal muscle ($\text{Muscle Density}$), serving as imaging biomarkers for bone mineral density and myosteatosis.




* **Population-Specific Reference Standards**: Embeds healthy adult distribution baselines (e.g., Kong et al., 2022) and clinical reference benchmarks to provide non-diagnostic, contextualized evaluation states.


* **Translational PACS Integration**:
* Automatically compiles an academic single-page **A4 standard three-line clinical report (PDF)**.


* Converts multi-tissue segmentations and structured findings into standard **DICOM Secondary Capture (SC)** objects.


* Provides one-click network export directly to institutional PACS servers via the standard **C-STORE** service.




* **Multilingual UI**: Seamless real-time switching between English, Chinese (中文), and Japanese (日本語).



### 2. App 2: CT-Based Diabetes Risk Prediction & Explainability Dashboard (`diabetes_predictor.py`)

* **Multivariable Risk Stratification**: Built on a retrospective development cohort using multivariable logistic regression, computing the probability of concurrent diabetes from age, bone HU, muscle HU, SMI, and the VAT/SAT ratio.


* **Dynamic Phenotype Alerts**: Evaluates myosteatosis based on BMI-stratified cutoffs, flagging disproportionate visceral fat accumulation and low muscle mass.


* **Dual SHAP Explanations**:
* **Individualized Dynamic Nomogram (SHAP Waterfall Plot)**: Decomposes patient-specific biomarker contributions, illustrating how each feature pushes the predicted log-odds above (red) or below (blue) the baseline.


* **Global Cohort Attribution (SHAP Beeswarm Plot)**: Displays overall feature impact distributions across the training cohort to ensure algorithmic transparency.





---

## 🖥️ Graphical User Interfaces

### App 1: Data Ingestion, Automated Segmentation & PACS Reporting

<img width="2880" height="3975" alt="1" src="https://github.com/user-attachments/assets/141d2f16-41ee-4e14-9885-3108dd31ddcb" />

*Figure 1: L3 automated localization, 2D nnU-Net 4-tissue segmentation overlay, regional quantitative evaluation table, and literature reference panel.*

---

### App 2: Diabetes Risk Prediction & Transparent SHAP Dashboard

<img width="2880" height="4592" alt="2" src="https://github.com/user-attachments/assets/b716f0cf-b10e-440e-ad53-807b0a835a3a" />

*Figure 2: Predicted diabetes probability, clinical phenotype prompts, individualized SHAP waterfall nomogram, and global beeswarm feature distributions.*

---

## 📁 Repository Structure

```text
.
├── L3_app.py                   # App 1: L3 body composition segmentation & PACS workflow
├── diabetes_predictor.py    # App 2: Diabetes risk prediction & SHAP dashboard
├── shap.csv                   # Reference cohort dataset for global SHAP visualization
├── 1.jpg                      # App 1 UI screenshot
├── 2.jpg                      # App 2 UI screenshot
├── Weights/                   # Model weight directory (configured separately)
│   ├── Dataset297_TotalSegmentator_total_3mm_1559subj/
│   └── nnUNetTrainer__nnUNetPlans__2d/
│       └── fold_all/
│           └── checkpoint_final.pth
├── requirements.txt           # Python dependency specifications
└── README.md                  # Documentation

```

---

## ⚙️ Environment Setup & Installation

It is recommended to run the suite inside a clean Conda virtual environment:

```bash
# 1. Create and activate a Conda environment
conda create -n ct-bodycomp python=3.10 -y
conda activate ct-bodycomp

# 2. Clone this repository
git clone [https://github.com/blasthao/L3-CT-Body-Composition-Profiling-Diabetes-Risk-Prediction].git
cd CT-L3-BodyComposition-Suite

# 3. Install PyTorch (select the appropriate CUDA version for your GPU)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Install scientific computing, UI, and explainability libraries
pip install streamlit numpy pandas matplotlib SimpleITK nibabel nilearn scikit-learn shap psutil

# 5. Install segmentation frameworks
pip install nnunetv2 totalsegmentator

# 6. Install medical imaging and PDF generation dependencies
pip install pydicom pynetdicom reportlab PyMuPDF

```

---

## 📦 Model Weights Configuration

Before running App 1, ensure the required deep learning weights are placed in the `Weights` folder (or specify custom paths in the application sidebar):

1. **TotalSegmentator Weights**:
* Official pretrained task: `Dataset297_TotalSegmentator_total_3mm_1559subj`
* Target path: `Weights/Dataset297_TotalSegmentator_total_3mm_1559subj`


2. **2D nnU-Net Segmentation Weights**:
* Trained model checkpoint: `checkpoint_final.pth` along with `dataset.json` and `plans.json`
* Target path: `Weights/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth`



---

## 🚀 Running the Applications

### Launch App 1 (Segmentation & PACS Integration)

```bash
streamlit run L3_app.py

```

1. Open your browser and navigate to `http://localhost:8501`.
2. Verify the model paths and PACS server connection parameters in the sidebar (click **"Test Connection (C-ECHO)"** to verify network connectivity).


3. Upload an abdominal NCCT DICOM ZIP archive or `.nii.gz` file, and specify patient height, sex, and age.


4. Click **"Start Automated Analysis"** to execute L3 localization, nnU-Net inference, metric calculation, and image rendering.


5. Download the complete output ZIP bundle (containing masks, metrics, and the A4 PDF report), or click **"Send Generated Images & PDF Report to PACS"** to archive the results.



### Launch App 2 (Diabetes Risk Prediction & SHAP Dashboard)

```bash
streamlit run diabetes_predictor.py --server.port 8502

```

1. Open your browser and navigate to `http://localhost:8502`.
2. Input patient body composition metrics extracted from App 1 (BMI, Age, Muscle HU, SMI, VAT/SAT Ratio, Cancellous Bone HU) using the sidebar sliders.


3. View the predicted probability, threshold risk classification, and clinical interpretation prompts in real time.


4. Analyze individual and global feature attributions via the interactive SHAP waterfall and beeswarm plots.



---

## 📚 Clinical Reference Guidelines & Literature

The reference distributions and clinical prompts implemented across both applications are grounded in the following peer-reviewed studies:

* **Cancellous Bone Attenuation (Bone HU)**:
* *Praveen AD, et al. CT image-based biomarkers for opportunistic screening of osteoporotic fractures: A systematic review and meta-analysis. Osteoporos Int. 2024;35(6):971-996.*

* *Pickhardt PJ, et al. Opportunistic screening for osteoporosis using abdominal computed tomography scans obtained for other indications. Ann Intern Med. 2013;158(8):588-595.*



* **Adipose Indices (SATI & VATI)**:
* *Kong M, et al. Assessing visceral obesity and abdominal adipose tissue distribution in healthy populations based on computed tomography: A large multicenter cross-sectional study. Front Nutr. 2022;9:871697.*



* **Skeletal Muscle Index (SMI)**:
* *Kong M, et al. Defining reference values for low skeletal muscle index at the L3 vertebra level based on computed tomography in healthy adults: A multicentre study. Clin Nutr. 2022;41(2):396-404.*

* *Prado CM, et al. Prevalence and clinical implications of sarcopenic obesity in patients with solid tumours of the respiratory and gastrointestinal tracts: a population-based study. Lancet Oncol. 2008;9(7):629-635.*



* **Visceral-to-Subcutaneous Ratio (VAT/SAT Ratio)**:
* *Emamat H, et al. The association between the visceral to subcutaneous abdominal fat ratio and the risk of cardiovascular diseases: a systematic review. BMC Public Health. 2024;24:1827.*



* **Skeletal Muscle Attenuation (Muscle HU / Myosteatosis)**:
* *Garcia-Diez AI, et al. Myosteatosis: diagnostic significance and assessment by imaging approaches. Quant Imaging Med Surg. 2024;14(11):7937-7957.*

* *Martin L, et al. Cancer cachexia in the age of obesity: skeletal muscle depletion is a powerful prognostic factor, independent of body mass index. J Clin Oncol. 2013;31(12):1539-1547.*




---

## ⚖️ Disclaimer

The software, algorithms, and calculated metrics provided in this repository are intended **strictly for scientific research, technical evaluation, and opportunistic screening feasibility studies**[cite: 14, 15]. Computed outputs, reference prompts, and predicted probabilities **do not constitute definitive clinical diagnoses or treatment recommendations**[cite: 14, 15]. In clinical practice, all imaging findings must be reviewed and interpreted by certified medical professionals in conjunction with patient-specific signs, symptoms, and laboratory data[cite: 14, 15].

---

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE). Please adhere to the individual licenses and terms of use of the underlying TotalSegmentator and nnU-Net frameworks when deploying pretrained weights.
