import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box 3"] #隐藏状态Y
n_states = len(states)              #隐藏状态数
observations = ["red", "white"]     #观测状态X
n_observations = len(observations)  #观测状态数
start_probability = np.array([0.2, 0.4, 0.4])   #初始状态概率π
transition_probability = np.array([ #状态转移矩阵A
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])
emission_probability = np.array([   #发射概率表B
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])
#由上述可确定模型 λ = (A,B,π)
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_ = emission_probability
#由确定的马尔可夫模型，给定观测序列，确定隐藏状态
seen = np.array([[0,1,0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print(box)


states = ["box 1", "box 2", "box 3"]
obs = ['red', 'white']
n_states = len(states)
m_obs = len(obs)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.001)
X2 = np.array([
    [0, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 1]
])
model2.fit(X2)      #由数据集X2来训练HiddenMarkov模型的相关参数
print("π")          # 同时由于HMM model.fit使用EM算法，故可能需要进行多次fit操作以提升准确度
print(model2.startprob_)
print("A")
print(model2.transmat_)
print("B")
print(model2.emissionprob_)
print("Score")
print(model2.score(X2))
