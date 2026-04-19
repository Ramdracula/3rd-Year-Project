# 3rd-Year-Project: Inkjet Print Defect Detection Dataset Pipeline

This repository contains the image capture, quality control, preprocessing, and manual labelling tools used for an inkjet print defect detection project.

The project investigates how controllable inkjet printing parameters influence visible print defects. Printed test patterns are captured using a Raspberry Pi camera system, stored with metadata, cleaned using human quality control, optionally cropped, and labelled using a multi-label defect taxonomy.

---

## Project Overview

The workflow is designed to support:

1. Controlled image capture of printed samples.
2. Storage of each image with matching process metadata.
3. Human quality control of captured images.
4. Cropping/preprocessing of image data.
5. Manual defect labelling using a standard taxonomy.
6. Later factor analysis and machine learning.

The main process variables are:

- DPI
- Pulse width
- Voltage
- Printhead height
- Print darkness
- Conveyor speed

The printed patterns are:

- `PATTERN001` — stripe pattern
- `PATTERN002` — QR code pattern
- `PATTERN003` — star pattern

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
└── README.md
