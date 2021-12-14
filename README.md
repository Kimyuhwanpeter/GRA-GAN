# Style Transfer of Gender, Race, and Age by GRA-GAN Based on Channel-wise and Multiplication-based Information Fusion of Encoder and Decoder Features

## Introduction

**Any work that uses the provided pretrained network must acknowledge the authors by including the following reference**

    Yu Hwan Kim, Se Hyun Nam, Seung Baek Hong, and Kang Ryoung Park, “Style Transfer of Gender, Race, and Age by GRA-GAN Based on Channel-wise and Multiplication-based Information Fusion of Encoder and Decoder Features,”  in submission 

<br>

## Implementation
* Python >= 3.5
* Tensorflow >= 2.1.0
* Window 10 or Linux
* Follow the "FLAGS" in the Gender_age_model_3_ver3.py

## Result
* Morph and AFAD (Age)

![Figure 1](https://github.com/Kimyuhwanpeter/GRA-GAN/blob/main/FIgure%201.png)
<br/>

* Morph and AFAD (Race and gender)

![Figure 2](https://github.com/Kimyuhwanpeter/GRA-GAN/blob/main/Figure%202.png)
<br/>

* Samples of training graphs (GRA-GAN, average the loss per step, it's not implemented in this code (if you want to make this graphs then you should add the average loss code).)
![Figure 3](https://github.com/Kimyuhwanpeter/GRA-GAN/blob/main/Figure%203.png)
<br/>

## Model weights (AFAD and Morph)
* [AFAD-M (16~39) and Morph-F (40 ~ 63)](https://drive.google.com/drive/folders/1wbuFFcIIgRBvfqRyeQDfufrJQlQiYm_C?usp=sharing)
* [AFAD-M (40~63) and Morph-F (16 ~ 39)](https://drive.google.com/drive/folders/1xwnCbq413JS7nuHBdKOd0AUDqzaePvv5?usp=sharing)
* [AFAD-F (40~63) and Morph-M (16 ~ 39)](https://drive.google.com/drive/folders/1xtUw5y-zevqtIbquOzS7a0huC9mlMdT7?usp=sharing)
* [AFAD-F (16~39) and Morph-M (40 ~ 63)](https://drive.google.com/drive/folders/11V1werP3BjBnUp5fCRv0_kUCIdT0DZFg?usp=sharing)
