### 目的 :
###### 使用來自600位不同講者的語音資料(已經過特徵工程)，
###### 訓練可以識別特定講者的聲音的自注意力模型。

### 訓練資料 : 
###### 約70000筆資料，每一筆資料包含:
###### 講者的姓名(編號)，以及一段語音，是由一串音素(phoneme)構成的序列，
###### 每個音素已經由課程助教，使用特徵工程轉換成40維的向量。

### 分析方法 : 
###### 建立多頭、多層的 self attention 模型，進行分類任務。

### 我的工作 : 
###### 改善模型架構: 使用attention pooling 
###### conformer
###### 

### 備註 :
###### attention pooling 參考資料 : 
###### Self-Attention Encoding and Pooling for Speaker Recognition
###### (Pooyan Safari, Miquel India and Javier Hernando, 2020)
###### link : https://arxiv.org/pdf/2008.01077v1
###### 未來加強學習的方向，是語音處理的相關特徵工程。
### 助教原檔:
###### origin_version.ipynb
