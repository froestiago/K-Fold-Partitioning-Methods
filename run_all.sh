#!/usr/bin/env bash
conda env create -f environment_file.yml 
conda activate splitters-env
python main.py hp-search
python main.py hp-search -s
python main.py n-clusters-estimate
python main.py n-clusters-estimate -a
python main.py true-estimate
python main.py true-estimate -s
python main.py true-estimate -a
python main.py compare-splitters
python main.py compare-splitters -s
python main.py compare-splitters -a