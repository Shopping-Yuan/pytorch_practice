### 專案介紹
###### OpenAI公司提供的Gym庫，有不同的虛擬環境供使用者進行強化學習的練習
###### LunarLander是其中的一個環境，使用者藉由reinforcement learning 的技術，
###### 讓訓練出的agent可以正確地駕駛登陸艇降落在星球表面。

![image](https://github.com/Shopping-Yuan/ML_Project/blob/Shopping_vscode_branch/Reinforcement_Learning/LunarLander.png)

### 方法 : 
###### 

### 我的工作 :
###### 1.在助教的原檔案當中，agent將目前的狀態資訊輸入神經網路，由輸出結果從四個動作中依機率選擇一個，
######   重複動作直到完整地跑完數次降落過程後，以過程中每次選擇的獎勵與選取特定動作的機率計算損失函數。
######   由損失函數，更新神經網路在給定狀態資訊的情況下，選擇四個動作的個別機率。
######   經過足夠的更新後，agent將會充分探索不同的狀態資訊下，不同動作的獎勵，並且將最優選擇的機率最大化。

######   我的第一個檔案rocket_landing.ipynb，僅對助教的檔案作微幅修改，修正agent每次作決定後的獎勵，
######   讓每個決定的獎勵不只包含當下，也包含後續動作的一部分獎勵，
######   如此一來越早的決定對結果影響越大，也符合真實的情況。

###### 2.實作deep Q learning

### 參考資料
###### 

### 助教原檔 : 
###### origin_version.ipynb
