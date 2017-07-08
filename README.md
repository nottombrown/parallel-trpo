# parallel-trpo

A parallel implementation of Trust Region Policy Optimization on environments from OpenAI gym

Tested with python 3.5 and Tensorflow 1.2.0

How to run:
```
# This just runs a simple training on Reacher-v1.
python main.py

# For the commands used to recreate results, check trials.txt

```
Parameters:
```
--task: what gym environment to run on
--timesteps_per_batch: how many timesteps for each policy iteration
--n_iter: number of iterations
--gamma: discount factor for future rewards_1
--max_kl: maximum KL divergence between new and old policy
--cg_damping: damp on the KL constraint (ratio of original gradient to use)
--num_threads: how many async threads to use
--monitor: whether to monitor progress for publishing results to gym or not
```
