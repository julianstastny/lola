"""The main scripts for running different scenarios."""

import click
import time
import datetime
import yaml

from lola import logger
import matplotlib.pyplot as plt
import pdb

from lola.envs import *


@click.command()
# Experiment parameters
@click.option("--exp_name", type=str, default="IPD",
              help="Name of the experiment (and correspondingly environment).")
@click.option("--num_episodes", type=int, default=None,
              help="Number of episodes.")
@click.option("--trace_length", type=int, default=None,
              help="Lenght of the traces.")
@click.option("--exact/--no-exact", default=True,
              help="Whether to run the exact version of LOLA.")
@click.option("--pseudo/--no-pseudo", default=False,
              help="Only used with exact version of LOLA.")
@click.option("--grid_size", type=int, default=3,
              help="Grid size of the coin game (used only for coin game).")
@click.option("--trials", type=int, default=1, help="Number of trials.")
@click.option("--seed", type=int, default=None, help="Random Seed.")
@click.option("--run_id", type=int, default=0, help="For the saving path.")
@click.option("--deploy_saved/--no-deploy_saved", default=False,
              help="Evaluate saved models rather than learn new ones.")


# Learning parameters
@click.option("--lola/--no-lola", default=True,
              help="Add the crazy LOLA corrections to the computation.")
@click.option("--opp_model/--no-opp_model", default=False,
              help="Whether to model opponent or use true parameters "
                   "(use only for coin game).")
@click.option("--mem_efficient/--no-mem_efficient", default=True,
              help="Use a more memory efficient corrections method.")
@click.option("--lr", type=float, default=None,
              help="Learning rate for Adam optimizer.")
@click.option("--lr_correction", type=float, default=1,
              help="Learning rate for corrections.")
@click.option("--batch_size", type=int, default=None,
              help="Number of episodes to optimize at the same time.")
@click.option("--bs_mul", type=int, default=1,
              help="Number of episodes to optimize at the same time")

# Policy parameters
@click.option("--simple_net/--no-simple_net", default=True,
              help="Use a simple policy (only for IPD and IMP).")
@click.option("--hidden", type=int, default=32,
              help="Size of the hidden layer.")
@click.option("--num_units", type=int, default=64,
              help="Number of units in the MLP.")
@click.option("--reg", type=float, default=0.,
              help="Regularization parameter.")
@click.option("--gamma", type=float, default=None,
              help="Discount factor.")
def main(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         trials, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
         num_units, reg, gamma, lola, opp_model, mem_efficient, seed, run_id,
         deploy_saved):

    if deploy_saved:
      self_play_payoffs_1 = []
      self_play_payoffs_2 = []
      cross_play_payoffs_1 = []
      cross_play_payoffs_2 = []
      # models_lst = ['./drqn/models/new-models-2/run_1/variables-3708',
      #             './drqn/models/new-models-2/run_2/variables-3689',
      #             './drqn/models/new-models-1/run_2/variables-3639',
      #             './drqn/models/new-models-3/run_1/variables-3786',
      #             './drqn/models/new-models-3/run_2/variables-3771',
      #             './drqn/models/new-models-4/run_1/variables-3810',
      #             './drqn/models/new-models-4/run_2/variables-3743',
      #             './drqn/models/new-models-5/run_1/variables-4168',
      #             './drqn/models/new-models-5/run_2/variables-2565',
      #             './drqn/models/new-models-6/run_1/variables-3480',
      #             './drqn/models/new-models-6/run_2/variables-3492',
      #             './drqn/models/new-models-7/run_1/variables-2582',
      #             './drqn/models/new-models-7/run_2/variables-3986',
      #             './drqn/models/new-models-8/run_1/variables-3562',
      #             './drqn/models/new-models-8/run_2/variables-3494'
      #             ]
      models_lst = ['./drqn/models/models1-3/run_1/variables-5000',
                    './drqn/models/models1-3/run_2/variables-3771']

      results_dict = {model_name: {} for model_name in models_lst}
      num_models = len(models_lst)
      for i in range(num_models):
        for j in range (num_models):
          model1 = models_lst[i]
          if i == j:
            sp1, sp2 = experiment(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
                       1, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
                       num_units, reg, gamma, lola, opp_model, mem_efficient, seed, run_id,
                       deploy_saved, path1=model1)
            self_play_payoffs_1.append(sp1)
            self_play_payoffs_2.append(sp2)
            results_dict[model1][model1] = (float(sp1), float(sp2))
          elif i < j:
            model2 = models_lst[j]
            cp1, cp2 = experiment(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
                                1, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
                                num_units, reg, gamma, lola, opp_model, mem_efficient, seed, run_id,
                                deploy_saved, path1=model1, path2=model2)
            cross_play_payoffs_1.append(cp1)
            cross_play_payoffs_2.append(cp2)
            results_dict[model1][model2] = (float(cp1), float(cp2))

      stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      fname = 'cg-lola-{}.yml'.format(stamp)
      with open(fname, 'w') as outfile:
        yaml.dump(results_dict, outfile)

      plt.scatter(self_play_payoffs_1, self_play_payoffs_2, label='self-play')
      plt.scatter(cross_play_payoffs_1, cross_play_payoffs_2, label='cross-play')
      plt.legend()
      plt.show()
    else:
      experiment(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
                 trials, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
                 num_units, reg, gamma, lola, opp_model, mem_efficient, seed, run_id,
                 deploy_saved)



def experiment(exp_name, num_episodes, trace_length, exact, pseudo, grid_size,
         trials, lr, lr_correction, batch_size, bs_mul, simple_net, hidden,
         num_units, reg, gamma, lola, opp_model, mem_efficient, seed, run_id,
         deploy_saved, path1=None, path2=None):
    # Sanity
    assert exp_name in {"CoinGame", "IPD", "IMP"}

    # Resolve default parameters
    if exact:
        num_episodes = 50 if num_episodes is None else num_episodes
        trace_length = 200 if trace_length is None else trace_length
        lr = 1. if lr is None else lr
    elif exp_name in {"IPD", "IMP"}:
        num_episodes = 600000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 1. if lr is None else lr
    elif exp_name == "CoinGame":
        num_episodes = 100000 if num_episodes is None else num_episodes
        trace_length = 150 if trace_length is None else trace_length
        batch_size = 4000 if batch_size is None else batch_size
        lr = 0.005 if lr is None else lr

    # Import the right training function
    if exact:
        assert exp_name != "CoinGame", "Can't run CoinGame with --exact."
        def run(env, save_path=None):
            from lola.train_exact import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  simple_net=simple_net,
                  corrections=lola,
                  pseudo=pseudo,
                  num_hidden=hidden,
                  reg=reg,
                  lr=lr,
                  lr_correction=lr_correction,
                  gamma=gamma)
    elif exp_name in {"IPD", "IMP"}:
        def run(env, save_path=None):
            from lola.train_pg import train
            train(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  gamma=gamma,
                  set_zero=0,
                  lr=lr,
                  corrections=lola,
                  simple_net=simple_net,
                  hidden=hidden,
                  mem_efficient=mem_efficient)
    elif exp_name == "CoinGame":
        if deploy_saved:
          from lola.deploy_cg import deploy
          def run(env):
            return deploy(env,
                  num_episodes=num_episodes,
                  trace_length=trace_length,
                  batch_size=batch_size,
                  bs_mul=bs_mul,
                  gamma=gamma,
                  grid_size=grid_size,
                  lr=lr,
                  corrections=lola,
                  opp_model=opp_model,
                  hidden=hidden,
                  mem_efficient=mem_efficient,
                  path1=path1, path2=path2
                  )
        else:
          def run(env, save_path=None):
              from lola.train_cg import train
              train(env,
                    num_episodes=num_episodes,
                    trace_length=trace_length,
                    batch_size=batch_size,
                    bs_mul=bs_mul,
                    gamma=gamma,
                    grid_size=grid_size,
                    lr=lr,
                    corrections=lola,
                    opp_model=opp_model,
                    hidden=hidden,
                    mem_efficient=mem_efficient,
                    path=save_path
                    )

    # Instantiate the environment
    if exp_name == "IPD":
        env = IPD(trace_length)
        gamma = 0.96 if gamma is None else gamma
    elif exp_name == "IMP":
        env = IMP(trace_length)
        gamma = 0.9 if gamma is None else gamma
    elif exp_name == "CoinGame":
        env = CG(trace_length, batch_size, grid_size)
        gamma = 0.96 if gamma is None else gamma

    # Run training
    if seed is None:
        assert trials==1, "If doing more than one trial, specify seed."
        # logger.configure(dir='logs/{}/no-seed-run{}'.format(exp_name, run_id))
        start_time = time.time()
        if deploy_saved:
          payoff_1, payoff_2 = run(env)
          return payoff_1, payoff_2
        else:
          run(env, save_path=f"./drqn/run_{run_id}")
        end_time  = time.time()

    else:
        for _seed in range(seed, seed + trials):
            logger.configure(dir='logs/{}/seed-{}'.format(exp_name, _seed))
            start_time = time.time()
            run(env, save_path=f"./drqn_{_seed}")
            end_time = time.time()


if __name__ == '__main__':
    results = main()
