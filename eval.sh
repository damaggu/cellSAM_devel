git clone https://github.com/JunMa11/NeurIPS-CellSeg/

pip install numba

# test
python neurips_test.py --debug 1 --model_path SAM_groundtruth_boxPrompt_everything_with_good_livecell_neurips_train.pth
python NeurIPS-CellSeg/baseline/compute_metric.py -s ./results1024/inferences -g ./evals/tuning/labels --gt_suffix _label.tiff --seg_suffix ".tiff"