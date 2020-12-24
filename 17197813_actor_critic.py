import argparse

import game
import graphical

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


training = False
EPISODES = 10000
LOSS_CLIPPING = 0.2
EPOCHS = 10
GAMMA = 0.99
BUFFER_SIZE = 2048
BATCH_SIZE = 256
NUM_ACTIONS = 160
NUM_BLOCKS = 80
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4


score_list = []
loss_list = []


DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))
move_list = []
for i in range(2):
    for j in range(8):
        for k in range(10):
            move_list.append([j, k, i == 1])


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = tf.keras.backend.sum(y_true * y_pred, axis=-1)
        old_prob = tf.keras.backend.sum(y_true * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)
        return -tf.keras.backend.mean(
            tf.keras.backend.minimum(
                r * advantage, tf.keras.backend.clip(
                    r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING
                ) * advantage
            ) + ENTROPY_LOSS * -(
                prob * tf.keras.backend.log(prob + 1e-10)
            )
        )
    return loss


class Agent:
    def __init__(self):
        self.critic = self.build_critic() if training else None
        self.actor = (
            self.build_actor() if training else self.build_play_actor()
        )
        self.episode = 0
        self.val = False
        self.reward = []
        self.gradient_steps = 0
        if training:
            self.dendy_rush = game.GameLogic()
            self.current_board = self.dendy_rush.board()
        else:
            speed = 10
            self.actor.load_weights(
                './weights/agent_weights.h5'
            )
            self.dendy_rush = graphical.Game(
                self.ai_callback, self.transition_callback,
                self.end_of_game_callback, speed
            )
            self.current_board = None
            self.dendy_rush.run()

    def build_actor(self):
        state_input = tf.keras.layers.Input(shape=(NUM_BLOCKS,))
        advantage = tf.keras.layers.Input(shape=(1,))
        old_prediction = tf.keras.layers.Input(shape=(NUM_ACTIONS,))

        x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = tf.keras.layers.Dense(
            NUM_ACTIONS, activation='softmax', name='output'
        )(x)

        model = tf.keras.models.Model(
            inputs=[state_input, advantage, old_prediction],
            outputs=[out_actions]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=LR),
            loss=[proximal_policy_optimization_loss(
                advantage=advantage, old_prediction=old_prediction
            )],
            experimental_run_tf_function=False
        )
        model.summary()

        return model

    def build_play_actor(self):
        state_input = tf.keras.layers.Input(shape=(NUM_BLOCKS,))

        x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = tf.keras.layers.Dense(
            NUM_ACTIONS, activation='softmax', name='output'
        )(x)

        model = tf.keras.models.Model(
            inputs=[state_input],
            outputs=[out_actions]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=LR),
            loss='mse', experimental_run_tf_function=False
        )
        model.summary()

        return model

    def build_critic(self):
        state_input = tf.keras.layers.Input(shape=(NUM_BLOCKS,))
        x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = tf.keras.layers.Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(
            inputs=[state_input], outputs=[out_value]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR), loss='mse')

        return model

    def get_action(self):
        self.current_board = self.convert_board()
        p = self.actor.predict([
            self.current_board.reshape(1, NUM_BLOCKS),
            DUMMY_VALUE, DUMMY_ACTION
        ])
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        actual_action = move_list[action]
        self.current_board = self.dendy_rush.board()
        return actual_action, action_matrix, p

    def reset_env(self):
        self.episode += 1
        self.val = False
        self.dendy_rush = game.GameLogic()
        self.current_board = self.dendy_rush.board()
        self.current_board = self.convert_board()
        self.reward = []

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def convert_board(self):
        letter_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, '#': 4}
        map_grid = [[]]
        i = 0
        res = isinstance(self.current_board, str)
        if(res):
            for c in self.current_board:
                if c == '\n':
                    map_grid.append([])
                    i += 1
                else:
                    map_grid[i].append(letter_to_num[c])
            self.current_board = np.array(map_grid)
        return self.current_board

    def get_batch(self):
        batch = [[], [], [], []]
        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, _ = self.dendy_rush.play(action)
            self.reward.append(reward)
            self.current_board = self.convert_board()
            tmp_batch[0].append(self.current_board)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.current_board = observation
            self.current_board = self.convert_board()

            if done:
                score_list.append(self.dendy_rush.score())
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = (
                            tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        )
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = (
            np.array(batch[0]), np.array(batch[1]), np.array(batch[2]),
            np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        )
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        if(training):
            while self.episode < EPISODES:
                obs, action, pred, reward = self.get_batch()
                obs, action, pred, reward = (
                    obs[:BUFFER_SIZE], action[:BUFFER_SIZE],
                    pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
                )
                old_prediction = pred
                obs = obs.reshape(-1, NUM_BLOCKS)
                pred_values = self.critic.predict(obs)

                advantage = reward - pred_values
                actor_loss = self.actor.fit(
                    [obs, advantage, old_prediction], [action],
                    batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                    verbose=False
                )
                critic_loss = self.critic.fit(
                    [obs], [reward], batch_size=BATCH_SIZE, shuffle=True,
                    epochs=EPOCHS, verbose=False
                )

                loss_list.append(actor_loss.history['loss'][-1])
                print(
                    f"Actor loss: {actor_loss.history['loss'][-1]:4.4f}\t"
                    f"Critic loss: {critic_loss.history['loss'][-1]:4.4f}\t"
                    f"Gradient Steps: {self.gradient_steps}"
                )
                self.gradient_steps += 1
            self.actor.save_weights('./weights/agent_weights.h5', True)

    def ai_callback(self, board, score, moves_left):
        self.current_board = board
        self.current_board = self.convert_board()

        p = self.actor.predict(self.current_board.reshape(1, NUM_BLOCKS))
        action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        actual_action = move_list[action]
        return actual_action

    def transition_callback(
        self, board, move, score_delta, next_board, moves_left
    ):
        pass

    def end_of_game_callback(self, boards, scores, moves, final_score):
        print(scores)
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='python 17197813_actor_critic --mode train/test'
    )
    parser.add_argument(
        '-m', '--mode', default='test', choices=['train', 'test'],
        help="The mode to run the AI in."
    )

    args = vars(parser.parse_args())

    if (args['mode'] == 'train'):
        training = True

    agent = Agent()
    if training:
        agent.run()
        loss_dataframe = pd.DataFrame(loss_list)
        score_dataframe = pd.DataFrame(score_list)
        loss_dataframe.plot(title='Agent Loss')
        plt.xlabel('Batch Episodes')
        plt.ylabel('Agent loss rate')
        score_dataframe.plot(title='Scores')
        plt.xlabel('Number of Games')
        plt.ylabel('Scores')
        plt.legend(score_dataframe.max())
        plt.show()
