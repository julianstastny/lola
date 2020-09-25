"""
Training funcion for the Coin Game.
"""
import os
import numpy as np
import tensorflow as tf
import pdb

from . import logger

from .corrections import *
from .networks import *
from .utils import *


def update(mainPN, lr, final_delta_1_v, final_delta_2_v):
  update_theta_1 = mainPN[0].setparams(
    mainPN[0].getparams() + lr * np.squeeze(final_delta_1_v))
  update_theta_2 = mainPN[1].setparams(
    mainPN[1].getparams() + lr * np.squeeze(final_delta_2_v))


def clone_update(mainPN_clone):
  for i in range(2):
    mainPN_clone[i].log_pi_clone = tf.reduce_mean(
      mainPN_clone[i].log_pi_action_bs)
    mainPN_clone[i].clone_trainer = \
      tf.train.GradientDescentOptimizer(learning_rate=0.1)
    mainPN_clone[i].update = mainPN_clone[i].clone_trainer.minimize(
      -mainPN_clone[i].log_pi_clone, var_list=mainPN_clone[i].parameters)


def deploy(env, *, num_episodes, trace_length, batch_size,
          corrections, opp_model, grid_size, gamma, hidden, bs_mul, lr,
          mem_efficient=True, path1='./models', path2='./models'):

  # Setting the training parameters
  batch_size = batch_size  # How many experience traces to use for each training step.
  trace_length = trace_length  # How long each experience trace will be when training

  y = gamma
  num_episodes = num_episodes  # How many episodes of game environment to train network with.
  load_model = False  # Whether to load a saved model.
  summary_step = 1
  n_agents = env.NUM_AGENTS
  total_n_agents = n_agents
  h_size = [hidden] * total_n_agents
  max_epLength = trace_length + 1  # The max allowed length of our episode.
  summary_len = 20  # Number of episodes to periodically save for analysis

  # tf.reset_default_graph()
  mainPN = []
  mainPN_step = []
  agent_list = np.arange(total_n_agents)
  for agent in range(total_n_agents):
    mainPN.append(
      Pnetwork('main' + str(agent), h_size[agent], agent, env,
               trace_length=trace_length, batch_size=batch_size, ))
    mainPN_step.append(
      Pnetwork('main' + str(agent), h_size[agent], agent, env,
               trace_length=trace_length, batch_size=batch_size,
               reuse=True, step=True))

  if not mem_efficient:
    cube, cube_ops = make_cube(trace_length)
  else:
    cube, cube_ops = None, None

  corrections_func(mainPN, batch_size, trace_length, corrections, cube)

  # saver1_path = os.path.join(path1, 'models-1/run_1/variables-1060.meta')
  # model1_path = os.path.join(path1, 'models-1/run_1/variables-1060')
  saver1_path = os.path.join(path1, 'models-1/run_2/variables-1060.meta')
  model1_path = os.path.join(path1, 'models-1/run_1/variables-1060')
  if path2 is not None:
  # saver2_path = os.path.join(path2, 'variables-1.meta')
    model2_path = os.path.join(path2, 'models-1/run_2/variables-1060')

  # create lists to contain total rewards and steps per episode
  jList = []
  rList = []
  aList = []

  total_steps = 0

  episodes_run = np.zeros(total_n_agents)
  episodes_run_counter = np.zeros(total_n_agents)
  episodes_reward = np.zeros((total_n_agents, batch_size))
  episodes_actions = np.zeros((total_n_agents, env.NUM_ACTIONS))

  pow_series = np.arange(trace_length)
  discount = np.array([pow(gamma, item) for item in pow_series])
  discount_array = gamma ** trace_length / discount
  discount = np.expand_dims(discount, 0)
  discount_array = np.reshape(discount_array, [1, -1])

  saver1 = tf.train.Saver(tf.trainable_variables())
  with tf.Session() as sess:
    # saver_1 = tf.train.import_meta_graph(saver1_path)
    # ckpt = tf.train.get_checkpoint_state('./drqn/run_1')
    # saver1.restore(sess, ckpt.model_checkpoint_path)
    saver1.restore(sess, model1_path)
    if path2 is not None:
      sess2 = tf.Session()
      sessions = [sess, sess2]
      saver1.restore(sess2, model2_path)
    else:
      sessions = [sess, sess]

    if not mem_efficient:
      sess.run(cube_ops)

    sP = env.reset()
    for i in range(num_episodes):
      episodeBuffer = []
      for ii in range(n_agents):
        episodeBuffer.append([])
      np.random.shuffle(agent_list)
      if n_agents == total_n_agents:
        these_agents = range(n_agents)
      else:
        these_agents = sorted(agent_list[0:n_agents])

      # Reset environment and get first new observation
      sP = env.reset()
      s = sP

      trainBatch0 = [[], [], [], [], [], []]
      trainBatch1 = [[], [], [], [], [], []]
      d = False
      rAll = np.zeros((4))
      aAll = np.zeros((env.NUM_ACTIONS * 2))
      j = 0

      lstm_state = []
      for agent in these_agents:
        episodes_run[agent] += 1
        episodes_run_counter[agent] += 1
        lstm_state.append(np.zeros((batch_size, h_size[agent] * 2)))

      while j < max_epLength:
        lstm_state_old = lstm_state
        j += 1
        a_all = []
        lstm_state = []
        for agent_role, agent in enumerate(these_agents):
          policy_sess = sessions[agent_role]
          a, lstm_s = policy_sess.run(
            [
              mainPN_step[agent].predict,
              mainPN_step[agent].lstm_state_output
            ],
            feed_dict={
              mainPN_step[agent].state_input: s,
              mainPN_step[agent].lstm_state: lstm_state_old[agent]
            }
          )
          lstm_state.append(lstm_s)
          a_all.append(a)

        a_all = np.transpose(np.vstack(a_all))

        s1P, r, d = env.step(actions=a_all)
        s1 = s1P

        total_steps += 1
        for agent_role, agent in enumerate(these_agents):
          episodes_reward[agent] += r[agent_role]

        for index in range(batch_size):
          r_pb = [r[0][index], r[1][index]]
          # Instead just record each agent's rewards
          rAll[0] += r_pb[0]
          rAll[1] += r_pb[1]
          # if r_pb[0] == 1 and r_pb[1] == 0:
          #     rAll[0] += 1
          # elif r_pb[0] == 0 and r_pb[1] == 1:
          #     rAll[1] += 1
          # elif r_pb[0] == 1 and r_pb[1] == -2:
          #     rAll[2] += 1
          # elif r_pb[0] == -2 and r_pb[1] == 1:
          #     rAll[3] += 1

        aAll[a_all[0]] += 1
        aAll[a_all[1] + 4] += 1
        s_old = s
        s = s1
        sP = s1P
        if d:
          break

      jList.append(j)
      rList.append(rAll)
      aList.append(aAll)

      episodes_run_counter[agent] = episodes_run_counter[agent] * 0
      episodes_actions[agent] = episodes_actions[agent] * 0
      episodes_reward[agent] = episodes_reward[agent] * 0

      if len(rList) % summary_len == 0 and len(rList) != 0:
        summary_step += 1
        print(total_steps, 'mean episode reward', np.mean(rList, 0) / batch_size)
        rlog = np.sum(rList[-summary_len:], 0)
        for ii in range(len(rlog)):
          logger.record_tabular('rList[' + str(ii) + ']', rlog[ii])
        logger.dump_tabular()
        logger.info('')
