### 目的 :
###### 訓練模型進行英翻中任務

### 資料集 : 
###### 39萬筆中英文的句對，來自於TED演講。

### 分析方法 : 
###### 使用Transformer，進行分類任務。

### 我的工作 : 
###### 1.建置transformer模型:
###### 未使用pytorch中self-attention層以及包含此結構的內建模塊，
###### 而是改寫scaled_dot_product_attention函數(無法處理padding mask)，
###### 結合輸入中對短句結尾的padding mask和decoder中的attention mask，
###### 完整實作transformer模型。
###### 2.實作beam search方法，期望增加輸出的品質。
###### 3.訓練中發現模型的輸出傾向使用疊字或標點延長句子的問題，
###### 與同伴討論後，認為是使用padding mask造成模型無法正確的斷句，
###### 但同伴發現如果直接使用無padding的輸入，則由於輸入過長不利於學習。
###### 最後我決定將訓練分為2個步驟，先以有padding mask的輸入訓練，
###### 最後再以無padding的輸入微調模型。
###### 4.使用2個頭，共6層的transformer結構。

### 結果 :
###### 1.我的decoder運行的速度顯著慢於使用內建的模塊，
###### 可能是由於我的scaled_dot_product_attention是以python而非C++實作，
###### 再加上以及其他可能的優化，造成了速度上的差距。
###### 在使用beam search方法進行驗證時，
###### 由於必須大量重複地利用decoder生成序列，
###### 速度上的差異會更加明顯。
###### 2.在訓練的第二步，經過3000個step後，重複輸出疊字或標點的狀況確實消失了。

### 資料集來源:
###### TED-Multilingual-Parallel-Corpus (by ajinkyakulkarni14)
###### link : https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus

### 參考資料
###### 1.助教原檔 : origin_version.ipynb (內附資料集連結)
###### 2.ChineseNMT (by hemingkx)
######   link : https://github.com/hemingkx/ChineseNMT/blob/master/README.md
###### 3.The Annotated Transformer (by harvardnlp)
######   link : https://nlp.seas.harvard.edu/2018/04/03/attention.html

