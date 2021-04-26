import argparse
import gym
import numpy as np


def value_iteration(env, discount, shaping=None, precision=1e-3, verbose=False):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    V = np.zeros(env.observation_space.n, dtype=np.float64)
    Q = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float64) # APREE
    iters = 0

    while True:
        iters += 1
        V_old = V.copy()

        try:                                                                            # APREE
            state_space = env.states()
            action_space = env.actions()
        except: 
            state_space = range(env.observation_space.n)
            action_space = range(env.action_space.n)

        for s in state_space:
            Q_values = [backup(env, discount, V, s, a, shaping) for a in action_space]
            Q[s] = Q_values.copy()                                                             # APREE
            V[s] = max(Q_values)
        
        if verbose: print(np.abs(V - V_old).max())
        if np.abs(V - V_old).max() <= precision:
            # return V
            policy = np.argmax(Q, axis=1)
            print(f'Value Iteration ran in {iters} iterations.')
            # to_print = np.reshape(policy.copy(),(11,11))
            # if verbose: print(f'Policy is\n{to_print}')
            return policy


def policy_iteration(env, discount, shaping=None, precision=1e-3, verbose=False):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0

    # For the sake of determinism, we start with the policy that always chooses action 0
    policy = np.zeros(env.observation_space.n, dtype=np.int32)
    iters = 0

    while True:
        iters += 1
        V_policy = policy_evaluation(env, discount, policy, shaping, precision, verbose)
        policy, stable = policy_improvement(env, discount, policy, V_policy, shaping, precision, verbose)
        to_print = np.reshape(policy.copy(),(11,11))
        if verbose: print(f'current policy is\n{to_print}')
        if stable:
            print(f'Policy Iteration ran in {iters} iterations.')
            return policy


def policy_evaluation(env, discount, policy, shaping=None, precision=1e-3, verbose=False):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    V = np.zeros(policy.shape, dtype=np.float64)

    while True:
        V_old = V.copy()

        try: state_space = env.states()
        except: state_space = range(env.observation_space.n)
        
        for s in state_space: 
            V[s] = backup(env, discount, V, s, policy[s], shaping)

        # if verbose: 
        #     print(np.abs(V - V_old).max())
        if np.abs(V - V_old).max() <= precision:
            return V


####################
# Helper functions #
####################


def policy_improvement(env, discount, policy, V_policy, shaping=None, precision=1e-3, verbose=False):
    policy_old = policy.copy()
    V_old = V_policy.copy()

    try: 
        state_space = env.states()
        action_space = env.actions()
    except: 
        state_space = range(env.observation_space.n)
        action_space = range(env.action_space.n)

    for s in state_space:
        Q_values = [backup(env, discount, V_policy, s, a, shaping) for a in action_space]
        policy[s] = np.argmax(Q_values)
        V_policy[s] = max(Q_values)

    stable = np.logical_or(
        policy == policy_old,
        np.abs(V_policy - V_old).max() <= precision,
    ).all()

    if verbose: print(np.abs(V_policy - V_old).max())
    # if stable: print(V_policy)
    return policy, stable


def backup(env, discount, V, state, action, shaping=None):
    try:
        next_states, rewards, dones, probs = env.model(state, action)
    except:
        transitions = env.P[state][action]
        probs = np.array([t[0] for t in transitions])
        next_states = np.array([t[1] for t in transitions], dtype=int)
        rewards = np.array([t[2] for t in transitions])
        dones = np.array([t[3] for t in transitions], dtype=int)
    
    if shaping is not None:
        shaped_rewards = []
        for idx, done in enumerate(dones):
            if done == 0:
                shaped_rewards.append(shaping['STEP'])
            else:
                if rewards[idx] == min(env.reward_range): # got minimum unshaped reward, i.e., lost
                    shaped_rewards.append(shaping['LOSE'])
                else:
                    shaped_rewards.append(shaping['WIN'])
        
        rewards = np.array(shaped_rewards)

    bootstraps = (1.0 - dones) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))
