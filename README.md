# Analytic-Continual-Test-Time-Adaptation-for-Multi-Modality-Corruption

This is the official implementation for [Analytic-Continual-Test-Time-Adaptation-for-Multi-Modality-Corruption](https://arxiv.org/abs/2410.22373)

### Initialize
1. Build your own environment through
```ubuntu
git clone https://github.com/FeeFee-1/Analytic-Continual-Test-Time-Adaptation-for-Multi-Modality-Corruption.git
cd MDAA
pip install -r requirements.txt
```

2. Prepare dataset
Follow [READ](https://github.com/XLearning-SCU/2024-ICLR-READ) download and generate your own corrupted datasets.

3. Prepare checkpoints
Follow the instruction in [READ](https://github.com/XLearning-SCU/2024-ICLR-READ) to download checkpoints for CAV-MAE.

Building on the previous step, download the [pre-trained models](https://drive.google.com/file/d/1Im6bs4UevadmY0pFWs3BWsgc2b30HA2o/view?usp=drive_link) and place them in the same file **./checkpoints**.

---
### Inference
Choose the corruption type and method you want to use, for example:
```python
python run_MDAA.py --tta-method 'MDAA' --corruption-modality 'audio' --batch-size 1 --MDAApretrained True
```

---
### Citation
If our work is useful for your research, please cite the following paper:
```
@article{zhang2024analytic,
  title={Analytic Continual Test-Time Adaptation for Multi-Modality Corruption},
  author={Zhang, Yufei and Xu, Yicheng and Wei, Hongxin and Lin, Zhiping and Zhuang, Huiping},
  journal={arXiv preprint arXiv:2410.22373},
  year={2024}
}
```

---
### Acknowledgements
The code is based on [READ](https://github.com/XLearning-SCU/2024-ICLR-READ) and [CAV-MAE](https://github.com/YuanGongND/cav-mae?tab=readme-ov-file#pretrained-models) licensed under Apache 2.0.
