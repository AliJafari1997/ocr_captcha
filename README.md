# ocr_captcha

clone this repository:
```
git clone https://github.com/AliJafari1997/ocr_captcha.git
```

This is a simple keras implementation for captcha recognition with CNN and LSTM.

changing directory to the main directory
```
cd captcha_ocr/src
```
download data from google drive:
```
!gdown  1TK-_hKb6MTnKuExthVBmCMGnXdN3nCah
```
making direcory for captcha dataset:
```
mkdir samples
```
unzip to the dataset to the defined folder
```
unzip -qq 'samples.zip' -d samples
```

implementing train.py module for calculating loss on train, validation, and test datasets.
```python
python train.py
```
