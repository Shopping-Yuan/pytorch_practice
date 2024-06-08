### 目的 :
###### 訓練模型，可以自動生成動漫風格的頭像。

### 資料集 : 
###### 來自Crypko，大約7萬張左右的動漫風格頭像。

### 方法 : 
###### 建立並訓練GAN模型。

### 我的工作 :
###### 1.使用gradient penalty方法改進GAN模型
###### 2.以W-GAN模型為基礎，實作SN-GAN模型(spectural normalization GAN)

### 參考資料
###### 1. gradient penalty : 
######    Improved Training of Wasserstein GANs
######    Ishaan Gulrajani et al. (2017)
######    link : https://arxiv.org/pdf/1704.00028
###### 2. spectural normalization for GAN
######    1.Spectural normalization for Generative adversarial networks
######    Takeru Miyato et al. (2018)
######    link : https://arxiv.org/pdf/1802.05957
######    2.Spectral norm regularization for improving the generalizability of deep learning.
######    Yuichi Yoshida and Takeru Miyato (2017)
######    link : https://arxiv.org/pdf/1705.10941
###### 3. SNGAN (by niffler92)
######    link : https://github.com/niffler92/SNGAN

###### 助教原檔 : origin_version.ipynb (內附資料集連結)
