### 專案介紹
###### OpenAI公司提供的Gym庫，有不同的虛擬環境供使用者進行強化學習的練習
###### LunarLander是其中的一個環境，使用者藉由reinforcement learning 的技術，
###### 讓訓練出的agent可以正確地駕駛登陸艇降落在星球表面。

![image](https://github.com/Shopping-Yuan/ML_Project/blob/Shopping_vscode_branch/Reinforcement_Learning/LunarLander.png)

### 方法 : 
###### 

### 我的工作 :
###### 1.policy gradient 微調策略
######   在助教的原檔案當中，agent將目前的狀態資訊輸入神經網路，由輸出結果從四個動作中依機率選擇一個，
######   重複動作直到完整地跑完數次降落過程後，以過程中每次選擇的獎勵與選取特定動作的機率計算損失函數。
######   由損失函數，更新神經網路在給定狀態資訊的情況下，選擇四個動作的個別機率。
######   經過足夠的更新後，agent將會充分探索不同的狀態資訊下，不同動作的獎勵，並且將最優選擇的機率最大化。

######   我的第一個檔案rocket_landing.ipynb，僅依助教提示對檔案作微幅修改，修正agent每次作決定後的獎勵，
######   讓每個決定的獎勵不只包含當下，也包含後續動作的一部分獎勵，
######   如此一來越早的決定對結果影響越大，也符合真實的情況。

###### 2.實作deep Q learning:
######   deep Q learning(DQN)是由Q learning結合神經網路結構而成，
######   Q-learning需要一個起始的，包含(狀態-動作-價值)的Q表，
######   並透過蒐集(狀態-動作-獎勵-下一狀態)的資訊獲得/更新每個狀態中，
######   不同動作的價值(假設學習策略可由馬爾可夫過程找出)，更新Q表。
######   DQN中，Q表為神經網路所取代，
######   透過蒐集足夠(狀態-動作-獎勵-下一狀態)的資訊更新Q值。
######   實作更新Q值的方式，
######   是藉由比較Q神經網路經由最小化
######   1.Q對目前狀態採取的動作的估值
######   2.Q對下一狀態預期最高獎勵的估值
######   二者之差，故在訓練過程中，會將Q化為二個神經網路(policy/target net)
######   每次更新其一(policy net)，數次更新後另一個網路(target net)再將前一個(policy net)的參數讀入。

###### 3.預訓練:
######   由於一開始的策略較差，運行中可能較難收集到較佳策略的資料。
######   而雖然deep Q learning估計動作價值，plicy gradient估計動作的機率，
######   但我的同伴提出，可以使用不含softmax層的policy gradient來作預訓練，
######   經過適當的參數調整(模型對學習率相當敏感)，有較好的結果。

### 參考資料
###### 1.李宏毅老師RL課程 :
###### 參考連結 : https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php (RL部分)
###### 2.An Introduction to Q-Learning: A Tutorial For Beginners (datacamp)
###### https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial

### 助教原檔 : 
###### origin_version.ipynb
