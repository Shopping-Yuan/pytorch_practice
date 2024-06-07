### 目的 :
###### 訓練模型進行英翻中任務

### 資料集 : 
###### 39萬筆中英文的句對，來自於TED演講。

### 分析方法 : 
###### 使用Transformer，進行分類任務。

### 我的工作 : 
###### 1.建置transformer模型:
###### 未使用pytorch中self-attention層以及包含此結構的內建模型，
###### 而是改寫scaled_dot_product_attention函數(無法處理padding mask)，
###### 結合輸入中對短句結尾的padding mask和decoder中的attention mask，
###### 完整實作transformer模型。
###### 2.實作beam search方法，期望增加輸出的品質。

### 備註 :
###### 
###### 

### 資料集來源:
###### TED-Multilingual-Parallel-Corpus (by ajinkyakulkarni14)
###### link : https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus

### 參考資料
###### 1.助教原檔 : origin_version.ipynb (內附資料集連結)
###### 2.ChineseNMT (by hemingkx)
######   link : https://github.com/hemingkx/ChineseNMT/blob/master/README.md
###### 3.The Annotated Transformer (by harvardnlp)
######   link : https://nlp.seas.harvard.edu/2018/04/03/attention.html

