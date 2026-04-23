# 3rd-Year-Project: Inkjet Print Defect Detection Pipeline

This repository contains the end-to-end data pipeline used for an inkjet print defect detection project, from controlled image capture to metadata analysis and classical image-based modelling.

The project studies how controllable inkjet printing parameters influence visible print defects. Printed test patterns are captured with a Raspberry Pi camera system, stored alongside process metadata, filtered through manual quality control, cropped for analysis, labelled using a standard defect taxonomy, and then analysed using both metadata-driven and image-feature-driven methods.

---

## Project Overview

The workflow supports:

1. Controlled image capture of printed samples.
2. Storage of each image with matching process metadata.
3. Manual quality control to remove unusable images.
4. Cropping and preprocessing to isolate the printed region.
5. Manual defect labelling using a multi-defect severity taxonomy.
6. Metadata-first factor analysis to identify which process variables are associated with defects.
7. Classical image-feature extraction and supervised modelling for defect prediction.

---

## Main Process Variables

- DPI
- Pulse width
- Voltage
- Printhead height
- Print darkness
- Conveyor speed

---

## Printed Patterns

- `PATTERN001` — stripe pattern
- `PATTERN002` — QR-style pattern
- `PATTERN003` — star pattern

---

## Defect Taxonomy

Each reviewed image can be labelled for the following defects:

- `dropout`
- `banding`
- `weak_print`
- `geometry_distortion`

Severity is recorded manually as:

- `0` = absent
- `1` = present / mild
- `2` = severe

Quality-control flags are handled separately before defect labelling.

---

## Repository Contents and Order of Use

```text
.
├── calibrate.py
├── calibration.json
├── capture_run.py
├── current_run.json
├── STEP_1_qc_review.py
├── STEP_2_crop_to_paper.py
├── STEP_3_exploratory_crop.py
├── STEP_4_label_defects.py
├── STEP_5_metadata_factor_analysis.py
├── STEP_6_extract_image_features.py
├── STEP_7_train_classical_image_models.py
└── README.md
