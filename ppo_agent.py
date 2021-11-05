import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input,Conv2D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import os
import time

# 속도 개선
tf.compat.v1.disable_eager_execution()
# 개발할 때 키기
# tf.config.experimental_run_functions_eagerly(True)

# -1 이면 cpu 0이면 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gpu = tf.config.experimental.list_physical_devices('GPU')
if len(gpu) > 0:
    print(f'GPUs {gpu}')
    try:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    except RuntimeError:
        pass

#############################################
learning_rate = 0.003
loss_clipping = 0.2
entropy_loss = 0.001

epoch = 10
gamma = 0.99
lmbda = 0.95


#############################################

class PPO_act_Network:
    def __init__(self, state_shape, action_size):
        self.action_size = action_size
        input_x = Input(state_shape)
        conv1 = Conv2D(16, (8, 8), strides=4, activation='relu')(input_x)
        conv2 = Conv2D(32, (4, 4), strides=2, activation='relu')(conv1)
        flatten = Flatten()(conv2)
        fc1 = Dense(256, activation='relu')(flatten)
        output = Dense(action_size, activation='softmax')(fc1)

        self.Model = Model(input_x, output)
        self.Model.compile(loss=self.ppo_loss, optimizer=RMSprop(learning_rate=learning_rate))
        self.Model.summary()

    def ppo_loss(self, y_true, y_pred):
        advantage = y_true[:, :1]
        action_prob = y_true[:, 1:1 + self.action_size]
        action = y_true[:, 1 + self.action_size:]

        prob = action * y_pred
        old_prob = action * action_prob

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantage
        p2 = K.clip(ratio, min_value=1 - loss_clipping, max_value=1 + loss_clipping) * advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = entropy_loss * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss


class PPO_critic_Network:
    def __init__(self, state_shape, action_size):
        input_x = Input(state_shape)
        old_v = Input(shape=(1,))

        conv1 = Conv2D(16,(8,8),strides=4,activation='relu')(input_x)
        conv2 = Conv2D(32,(4,4),strides=2,activation='relu')(conv1)
        flatten = Flatten()(conv2)
        fc1 = Dense(256, activation='relu')(flatten)
        output = Dense(1, activation=None)(fc1)

        self.Model = Model([input_x, old_v], output)
        self.Model.compile(loss=self.cri_loss(old_v), optimizer=RMSprop(learning_rate=learning_rate))

        self.Model.summary()

    def cri_loss(self, old_v):
        v = old_v

        def loss(y_true, y_pred):
            clipped = v + K.clip(y_pred - v, -loss_clipping, loss_clipping)

            value_loss_1 = (y_true - clipped) ** 2
            value_loss_2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.mean(K.maximum(value_loss_1, value_loss_2))

            return value_loss

        return loss


class PPO_Agent:

    def __init__(self, state_shape, action_size,mode):

        self.state_shape = state_shape
        self.action_size = action_size
        self.reset_buffer()
        self.epi = 0
        self.limit_count = 0
        self.count = 0
        self.dummy = np.zeros((32, 1))
        self.train_count = 0
        self.mode = mode

        # 모델 생성
        self.act_model = PPO_act_Network(self.state_shape, self.action_size)
        self.critic_model = PPO_critic_Network(self.state_shape, self.action_size)

        if(mode == 'test'):
            self.act_model.Model.load_weights("./_actor/")
            self.critic_model.Model.load_weights("./_critic/")

    # 메모리 초기화
    def reset_buffer(self):
        self.s = []
        self.a = []
        self.r = []
        self.s_ = []
        self.act_probs = []
        self.masks = []

    # 학습에 쓸 데이터 저장
    def add_buffer(self, state, action, reward, next_state, act_prob, done):

        self.s.append(state)
        self.a.append(action)
        self.r.append([reward])
        self.s_.append(next_state)
        self.act_probs.append(act_prob)
        mask = 1 - done
        self.masks.append([mask])


    def all_buffer_to_numpy(self):

        self.s = np.asarray(self.s)
        self.a = np.asarray(self.a)
        self.r = np.asarray(self.r)
        self.s_ = np.asarray(self.s_)
        self.act_probs = np.asarray(self.act_probs)
        self.masks = np.asarray(self.masks)

    # by https://arxiv.org/abs/1506.02438
    def get_advantages(self):
        deltas = self.r + gamma * self.masks * self.v_next - self.v
        gaes = deltas
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + self.masks[t] * gamma * lmbda * gaes[t + 1]

        target = gaes + self.v
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.vstack(gaes), np.vstack(target)

    def train(self,env):
        env.pause()
        self.all_buffer_to_numpy()

        self.v = np.asarray(self.critic_model.Model.predict_on_batch([self.s, self.dummy]))
        self.v_next = np.asarray(self.critic_model.Model.predict_on_batch([self.s_, self.dummy]))

        self.advantages, self.target = self.get_advantages()

        batch = self.s.shape[0]
        y_true = np.hstack([self.advantages, self.act_probs, self.a])
        print(" train log >> actor start")
        act_his = self.act_model.Model.fit(self.s, y_true, epochs=epoch, batch_size=batch, verbose=1)
        print(" train log >> actor clear")

        print(" train log >> critic start")
        cri_his = self.critic_model.Model.fit([self.s, self.v], self.target, epochs=epoch, verbose=1, batch_size=batch)
        print(" train log >> critic clear")

        self.save_model()

        self.reset_buffer()
        env.pause_switch = False

    def get_act(self, state):
        action_model_predict = self.act_model.Model.predict(state)[0]
        action = np.random.choice(self.action_size, p=action_model_predict)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1

        return action, action_onehot, action_model_predict

    def evaluate_get_act_(self, state):
        action_model_predict = self.act_model.Model.predict(state)[0]
        action = np.argmax(action_model_predict)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1

        return action_onehot

    def save_model(self):

        actor = "./_actor/"
        critic = "./_critic/"
        print("log >> save model start")
        # self.act_model.Model.save_weights(filepath=actor, overwrite=True, save_format="tf")
        # self.critic_model.Model.save_weights(filepath=critic, overwrite=True, save_format="tf")
        print("log >> save model fin")
