import numpy as np
rng = np.random.default_rng()

from typing import Tuple, Dict

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import japanize_matplotlib


class AgentArchetype(ABC):
    R = 5  # 基本報酬
    alpha = 0.1  # 学習率
    gamma = 0.9  # 減衰率
    lambda_1_initial = 0.5  # 初期 λ_1
    lambda_2_initial = 0.5  # 初期 λ_2
    bias_1_initial = 0  # 初期 b_1
    bias_2_initial = 0 # 初期 b_2
    possible_actions = ("Cooperate", "Defect")
    
    Q_initial = {
        ("Cooperate", "Cooperate"): 0.,
        ("Cooperate", "Defect"): 0.,
        ("Defect", "Cooperate"): 0.,
        ("Defect", "Defect"): 0.,
    
    #self.trustとself.estimated_trustを元に相手の判断を推定する。
    @abstractmethod
    def prediction(self):
        pass
    
    #　predictionを実行し、それをもとに"Cooperate", "Defect"のいずれかを決定する。
    @abstractmethod
    def decision(self):
        pass
    @abstractmethod
    def update_act(self):
        pass
    @abstractmethod
    def update_lambda_bias(self):
        pass

class Agent(AgentArchetype):
    def __init__(self, trust=0.5, estimated_trust=0.5):
        self.trust: float = trust
        self.estimated_trust: float = estimated_trust
        self.alpha: float = super().alpha
        self.gamma: float = super().gamma
        self.Q_1: Dict[Tuple[str, str], float] = super().Q_initial.copy() # 行動決定用 Q テーブル
        self.Q_2: Dict[Tuple[str, str], float] = super().Q_initial.copy() # lamda_1~2とbias_1~2の学習用Qテーブル
        self.lambda_1: float = super().lambda_1_initial
        self.lambda_2: float = super().lambda_2_initial
        self.bias_1: float = super().bias_1_initial
        self.bias_2: float = super().bias_2_initial
    
    
    #self.trustとself.estimated_trustを元に相手の判断を推定する。
    def prediction(self, eta: float) -> str:
        # 相手が黙秘する確率を見積もる。
        p_cooperate = 1 / (1 + np.exp(- (self.bias_1 + self.bias_2 + self.lambda_1 * self.trust + self.lambda_2 * self.estimated_trust)))

        if rng.random() < eta:
            return rng.choice(super().possible_actions)
        
        elif rng.random() < p_cooperate:
            return "Cooperate"
        
        else:
            return "Defect"
    
    
    # 推定した相手の行動とQ_1を元に自分の行動を決断する。
    def decision(self, eta: float) -> Tuple[str, str]:
        opponent_prediction = self.prediction(eta)
        
        if rng.random() < eta:
            return rng.choice(self.possible_actions), opponent_prediction
        
        else:
            max_action = max(self.possible_actions, key=lambda x: self.Q_1[(x, opponent_prediction)])
            if rng.random() < 1 - eta:
                return max_action, opponent_prediction
            else:
                return rng.choice(self.possible_actions), opponent_prediction
    
    def update_act(self, action, opponent_action, reward) -> None:
        # TD学習アルゴリズムを用いてQ_1を更新
        self.Q_1[(action, opponent_action)] *= (1 - self.alpha)
        self.Q_1[(action, opponent_action)] += self.alpha * (reward + self.gamma * max([self.Q_1[(x, opponent_action)] for x in self.possible_actions]))
    
    def update_lambda_bias(self, action, estimate, opponent_action, reward) -> None:
        # TD学習アルゴリズムを用いてQ_2を更新
        self.Q_2[(action, estimate)] *= (1 - self.alpha)
        self.Q_2[(action, estimate)] += self.alpha * (reward + self.gamma * self.Q_2[(action, estimate)])

        # lambdaとbiasの更新
        delta_Q = reward + self.gamma * self.Q_2[(action, estimate)] - self.Q_2[(action, estimate)]
        self.lambda_1 += self.alpha * self.lambda_1 * self.trust * (self.estimated_trust - self.Q_2[(action, estimate)]) * delta_Q
        self.lambda_2 += self.alpha * self.lambda_2 * self.estimated_trust * (self.trust - self.Q_2[(action, estimate)]) * delta_Q
        self.bias_1 += self.alpha * (self.estimated_trust - self.Q_2[(action, estimate)]) * delta_Q
        self.bias_2 += self.alpha * (self.trust - self.Q_2[(action, estimate)]) * delta_Q
        self.lambda_1 = np.clip(self.lambda_1, -1, 1)
        self.lambda_2 = np.clip(self.lambda_2, -1, 1)
        self.bias_1 = np.clip(self.bias_1, -10, 10)
        self.bias_2 = np.clip(self.bias_2, -10, 10)
    
    def act(self, action, estimate, opponent_action, up = True) -> float:
        # 報酬計算
        reward = super().R
        if action == "Cooperate" and opponent_action == "Cooperate":
            reward -= 2
        
        elif action == "Cooperate" and opponent_action == "Defect":
            reward -= 10
        
        elif action == "Defect" and opponent_action == "Cooperate":
            reward -= 0
        else:
            reward -= 5
        reward /= 5
        if up:
            self.update_act(action, opponent_action, reward)
            self.update_lambda_bias(action, estimate, opponent_action, reward)
        else:
            pass
        
        return reward

# 実施例
p1 = Agent(0.5, 0.5)
p2 = Agent(0.5, 0.5)

num = 1000 # 試行回数
result = [] # 選択のログ
p1_r, p2_r = [], [] # 獲得報酬ログ

for i in range(1, num+1):
    act1, est1 = p1.decision(0.5 * 1/(i*0.3))
    act2, est2 = p2.decision(0.5 * 1/(i*0.3))
    choices = (act1, act2)
    result.append(choices)
    p1_r.append(p1.act(act1, est1, act2))
    p2_r.append(p2.act(act2, est2, act1))

fig = plt.figure(figsize=(12*np.sqrt(2), 12))
ax = [fig.add_subplot(221 + i) for i in range(4)]

for n, a in enumerate(ax[0:2]):
    c_num = sum(np.array(result)[:, n] == "Cooperate")
    a.bar(("Cooperate", "Defect"), [c_num, num - c_num])
    a.text(0, num//2, c_num, fontsize=16)
    a.text(0.95, num//2, num - c_num, fontsize=16)
    a.set_title(f"囚人{n + 1}の選択")

for n, a, r in zip([1, 2], ax[2:], [p1_r, p2_r]):
    a.plot(np.arange(1, num + 1), np.cumsum(r))
    a.set_title(f"囚人{n}の報酬推移(総計: {sum(r): .2f})")

plt.show()