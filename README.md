# DTCE
This is the code of C&G paper: Real-time self-supervised tone curve estimation for HDR image.
## Environment Configuration
```conda create --name DTCE opencv imageio scipy pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 python=3.10 -c pytorch```
## Folder structure
```
├── data  
│   ├── testdata # testing data.  
│   ├── traindata # training data.  
│   └── result  
├── snapshots  
│   ├── Epoch399.pth #  A pre-trained snapshot
├── test.py # testing code
├── train.py # training code
├── model.py # DTCE network
├── dataloader.py
```
## Preparations
Download [train dataset](https://www.kaggle.com/datasets/landrykezebou/lvzhdr-tone-mapping-benchmark-dataset-tmonet)(LVZ-HDR Dataset.egg) and [test dataset](https://github.com/zhangn77/LTMN)(test_data).
## Test:
```python test.py ```
## Train:
```python train.py ```
## Contact
If you have any questions, please contact Xiyu Chen at chenxiyu@nimte.ac.cn or Jiayan Zhuang at zhaungjiayan@nimte.ac.cn.
