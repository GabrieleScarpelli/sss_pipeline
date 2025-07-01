# Side-Scan-Sonar Image Processing Pipeline

A Python package for processing and visualizing side-scan sonar data, starting from mission logs and raw sensor outputs.

## Description

Work In Progress.

## Requirements

- Python >= 3.7
- numpy
- pandas
- opencv-python
- matplotlib
- tqdm
- scipy
- PyYAML

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository and install in editable mode

```bash
git clone https://github.com/GabrieleScarpelli/sss_pipeline.git
cd sss_pipeline
pip install -e .
```

## Usage

Edit the config.yaml file according to your needs and run the command

```bash
run-sss-pipeline --config config.yaml
```
Make sure your main.py has a main() function that parses command-line args.

## Folder Structure

<pre>
sss_pipeline/
├── side_scan_sonar_pipeline/
│   ├── __init__.py
│   ├── io.py
│   ├── pipeline.py
│   └── processing.py
├── config.yaml
├── main.py
├── requirements.txt
├── setup.py
└── .gitignore
</pre>
