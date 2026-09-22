# A Stage-Wise Evaluation of YOLOv11s for PCB Surface Defect Detection
### Input, Backbone, Neck, Head, and Inference-Level Modifications

**Authors:** Yasmin Al Hendawy, Ramez Al-Masadeh, Dr. Ghaith Al Refai  
**Affiliation:** Department of Mechatronics and Artificial Intelligence Engineering, German Jordanian University (GJU), Amman, Jordan  
**Preprint / Publication:** Included in `paper/Defect_Detection_PCB.pdf`

---

## Overview

Automated Optical Inspection (AOI) of Printed Circuit Boards (PCBs) is critical in electronics manufacturing to eliminate manufacturing flaws such as broken traces, spurs, and shorts. While modern literature routinely adds attention modules or extra detection heads to boost benchmarks, this research conducts an exhaustive, stage-by-stage empirical ablation to evaluate whether such architectural additions genuinely outperform a well-tuned baseline.

This repository contains the model architectures, training configurations, and sliced inference scripts evaluated on the **DsPCBSD+ (Dataset of Printed Circuit Board Surface Defects)** benchmark.

---

## Dataset: DsPCBSD+

The dataset comprises 6 prominent industrial surface defect classes with extreme visual ambiguity and micro-scale dimensions:
1. **Hole Breakout (HB)**
2. **Missing Hole (MH)**
3. **Conductor Scratch (CS)**
4. **Conductor Foreign Object (CFO)**
5. **Spur (SP)**
6. **Short (SH)**

---

## Stage-Wise Ablation Results

| Stage | Experiment | Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Key Empirical Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | YOLOv11s (640px) | 50 | **0.809** | **0.790** | **0.839** | **0.499** | Strongest overall model; reference baseline |
| **Input** | EdgeMix Augmentation | Best | 0.823 | 0.766 | 0.827 | 0.488 | Sharpens boundaries; strongest non-baseline result |
| **Input** | High-Resolution (960px) | Best | 0.824 | 0.770 | 0.828 | 0.487 | Improved micro-feature resolution; high compute |
| **Input** | Rare-Class Oversampling | 28 | 0.810 | 0.779 | 0.810 | 0.451 | Oversampling led to localization degradation |
| **Backbone** | Coordinate Attention (CA) | 50 | 0.788 | 0.767 | 0.813 | 0.472 | Marginal gain on spurs; lagged baseline overall |
| **Backbone** | CBAM Attention | 50 | 0.802 | 0.765 | 0.811 | 0.473 | Channel+spatial attention showed no significant gain |
| **Head** | P2 Small-Object Head | 60 | 0.795 | 0.771 | 0.818 | 0.480 | High memory overhead without surpassing baseline |
| **Inference** | SAHI (Sliced Inference) | Test | 0.792 | 0.785 | 0.831 | 0.492 | Strong recall on tiny defects; higher latency |

---

## Key Takeaways

1. **The Scratch Paradox:** Conductor scratches (CS) and foreign objects (CFO) remain the most difficult defects across all models. Their limitation is caused by visual ambiguity and low-contrast surface texture, not lack of spatial resolution.
2. **Attention Limitations:** Standard attention modules (CBAM, Coordinate Attention) and auxiliary heads (P2) added parameter complexity but failed to surpass an optimized YOLOv11s baseline.
3. **EdgeMix Domain Augmentation:** Boundary-emphasizing spatial filtering proved the most competitive intervention, proving that domain-tailored input preprocessing yields greater practical utility than architectural bloat.

---

## Repository Layout

```text
├── paper/
│   └── Defect_Detection_PCB.pdf           # 7-page IEEE research paper
├── models/
│   ├── yolo11s_ca.yaml                    # Coordinate Attention backbone
│   ├── yolo11s_cbam.yaml                  # CBAM module integration
│   ├── yolo11s_p2.yaml                    # Dedicated P2 micro-defect head
│   └── yolo11s_p2_CA.yaml                 # P2 head + Coordinate Attention
├── scripts/
│   ├── test_sahi_100ep.py                 # SAHI sliced inference pipeline
│   ├── train_expB_yolo11s_cbam_oversampled.py # CBAM training script
│   ├── train_coordinate_attention.py      # CA training script
│   └── add_cbam_to_ultralytics.py         # PyTorch CBAM layer injector
└── README.md
```

---

## Citation

If you use this work, please cite:
```bibtex
@article{alhendawy2026yolov11pcb,
  title={A Stage-Wise Evaluation of YOLOv11s for PCB Surface Defect Detection: Input, Backbone, Neck, Head, and Inference-Level Modifications},
  author={Al Hendawy, Yasmin and Al-Masadeh, Ramez and Al Refai, Ghaith},
  journal={Department of Mechatronics and Artificial Intelligence Engineering, German Jordanian University},
  year={2026}
}
```
---

## Quickstart & Evaluation

The repository includes pre-trained weights (`weights/best.pt`), the dataset schema (`data/data.yaml`), and the test image split (`data/test/`) for immediate evaluation.

### 1. Installation
```bash
git clone https://github.com/RamezM2004/yolov11-pcb-defect-detection.git
cd yolov11-pcb-defect-detection
pip install -r requirements.txt
```

### 2. Run SAHI Sliced Inference on Test Set
```bash
python scripts/test_sahi_100ep.py
```

### 3. Download Full Training Dataset (Optional)
To download the complete DeepPCB / DSPCBSD training set from Roboflow Universe:
```bash
python scripts/download_dataset.py
```
