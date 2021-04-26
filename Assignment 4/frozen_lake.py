import numpy as np
import pandas as pd
import gym
from gym_classics.dynamic_programming import policy_iteration, value_iteration
from time import perf_counter


def epsilon_greedy(env, Q, state, epsilon):
    if np.random.rand() < epsilon: action = env.action_space.sample()
    else: action = np.argmax(Q[state])

    return action


def Q_learning(env, gamma=0.99, start_epsilon=0.3, epsilon_decay=0.1, epsilon_floor=0, epsilon_decay_speed = 100, \
                base_alpha=0.6, alpha_decay=0.95, alpha_decay_speed=100, tol=0.0001, num_episodes=10000, verbose=False):
    
    stats = pd.DataFrame(columns=['size','episode','alpha','epsilon','alpha decay','epsilon decay',\
        'gamma','score','result','time','num steps','max value change', 'policy'])
    this_episode_stats = {}

    # Hyperparameters for Q-Learning
    gamma = gamma
    epsilon = start_epsilon
    alpha = base_alpha

    # Our Q-function is a numpy array
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    converged = False

    for episode in range(num_episodes):
        old_Q = Q.copy()

        this_episode_stats['size'] = env.observation_space.n
        this_episode_stats['episode'] = episode
        this_episode_stats['alpha'] = alpha
        this_episode_stats['epsilon'] = epsilon
        this_episode_stats['alpha decay'] = alpha_decay
        this_episode_stats['epsilon decay'] = epsilon_decay
        this_episode_stats['gamma'] = gamma

        state = env.reset()
        done = False
        total_reward = 0
        num_steps = 0

        if episode > 0:
            if episode % epsilon_decay_speed == 0 and epsilon > epsilon_floor: 
                epsilon *= epsilon_decay
            if episode % alpha_decay_speed == 0:
                if alpha_decay == 'harmonic': 
                    alpha = base_alpha/(1+episode/alpha_decay_speed)                # this method satisfies the convergence qualification
                else:
                    assert type(alpha_decay) is float, 'Check alpha decay method.'
                    alpha *= alpha_decay                                            # but this method is faster and gets very good results

        t0 = perf_counter()
        while not done:
            num_steps += 1
            # Select action from ε-greedy policy
            action = epsilon_greedy(env=env, Q=Q, state=state, epsilon=epsilon)

            # Step the environment
            next_state, reward, done, _ = env.step(action)

            if (size == 32 and num_steps > 10000): 
                done = True
                reward = -1

            total_reward += reward

            # Q-Learning update:
            # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
            target = reward - Q[state, action]
            if not done:
                target += gamma * np.max(Q[next_state])
            
            else:
                if reward == 1: result = 1
                else: result = 0
            
            Q[state, action] = Q[state, action] + alpha * target
            state = next_state


        max_value_change = np.max(np.abs(Q - old_Q))
        this_episode_stats['score'] = total_reward
        this_episode_stats['time'] = perf_counter() - t0
        this_episode_stats['num steps'] = num_steps
        this_episode_stats['result'] = result
        this_episode_stats['max value change'] = max_value_change
        this_episode_stats['average Q'] = np.average(Q)
        this_episode_stats['policy'] = str(np.argmax(Q,axis=1))

        stats = stats.append(pd.DataFrame(this_episode_stats, index=[episode]))
        win_pct = np.sum(stats['result'][-100:])

        if verbose and episode % 100 == 0:
            print(f'Episode #{episode}\tReward:\t{round(total_reward,0)}\tResult:\t{result}\tWin %:\t{win_pct}%\tEpsilon:\t{round(epsilon,3)}\tAlpha:\t{round(alpha,3)}\tValue Change:\t{round(max_value_change,4)}\t\tAvg Q:\t{np.average(Q)}\tNum Steps:\t{num_steps}')

        # check for convergence
        if episode > 1000:
            if max(stats['max value change'][-10:]) < tol:
                print('Q-Learning converged at episode {}.'.format(episode))
                converged = True
                break
    
    if not converged: print('Q-Learning did not converge in {} episodes.'.format(episode))
    final_policy = np.argmax(Q,axis=1)
    values = pd.DataFrame(data=Q)

    return final_policy, stats, values


def test_final_policy(env, policy=None, num_episodes=100, verbose=False):
    reward_log = []
    win_log = []
    step_log = []

    t0 = perf_counter()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        num_steps = 0
        
        while not done:
            num_steps += 1
            try:
                action = policy[state]
            except:
                print('Warning, performed a random action where there was supposed to be a policy.')
                action = np.random.randint(low=0, high=env.action_space.n)

            next_state, reward, done, _ = env.step(action)
            
            if done:
                if reward == 1: win_log.append(1)
                else: win_log.append(0)
            
            total_reward += reward
            state = next_state

            if num_steps > 2000000:
                if verbose: print(f'Episode {episode}: too many steps.')
                done=True

        reward_log.append(total_reward)
        step_log.append(num_steps)
        if verbose: print(total_reward)

    t1 = perf_counter()
    
    print('{} wins out of {} episodes: {}% win rate. Average reward: {} (+/-{}). Time: {} seconds. Average steps: {}'.format(np.sum(win_log), num_episodes, 
                                                    round(100*np.sum(win_log)/num_episodes,4), 
                                                    round(np.average(reward_log),4),
                                                    round(np.std(reward_log),4),
                                                    round(t1-t0,4),
                                                    np.average(step_log)))
    
    return win_log, reward_log


def run_experiment(size=4, penalty=-0.001):
    np.random.seed(0)
    
    env = gym.make('FrozenLake-v0')
    # p is the probability that any square is frozen
    lake = gym.envs.toy_text.frozen_lake.generate_random_map(size=size, p=0.9)
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(desc=lake, is_slippery=True, step_penalty=penalty)
    env = env.unwrapped
    
    env.render()
    env.seed(0)

    solvers = ['Value Iteration','Policy Iteration','Q-Learning']
    policies = {}

    def hyperparam_study(gamma=0.99, base_alpha=0.6, alpha_decay=0.95, start_epsilon=0.3, epsilon_decay=0.2, param=None, val=None):
        print(f'\nResults for {param}={val}')
        t0 = perf_counter()
        policy, stats, values = Q_learning(env=env, gamma=gamma, base_alpha=base_alpha, alpha_decay=alpha_decay, alpha_decay_speed=100, \
            start_epsilon=start_epsilon, epsilon_decay=epsilon_decay, epsilon_floor=0.01, num_episodes=20000, verbose=False)

        print(f'Finished {solver} in {round((perf_counter()-t0)/60,2)} minutes.')
        print('Now testing final policy.')
        _, _ = test_final_policy(env=env, policy=policy, verbose=True)
        policies[solver] = policy

        assert solver == 'Q-Learning', 'Check solver, only Q-Learning accepted'
        with pd.ExcelWriter(f'Excel/frozenlake_{size}.xlsx',mode='a') as writer:
                    stats.to_excel(writer, sheet_name=f'Stats, {param}={val}')
                    values.to_excel(writer, sheet_name=f'Values, {param}={val}')

    for solver in solvers:
        print(f'======={solver} on size {size}=======')
        if solver == 'Policy Iteration':
            for gamma in [0.8, 0.9, 0.95, 0.99]:
                print(f'gamma={gamma}')
                policy = policy_iteration(env=env, discount=gamma, precision=0.0001, verbose=True) # precision and gamma are VERY VERY important for this
                _ = test_final_policy(env=env, policy=policy, verbose=True)
        elif solver == 'Value Iteration':
            for gamma in [0.8, 0.9, 0.95, 0.99]:
                print(f'gamma={gamma}')
                policy = value_iteration(env=env, discount=gamma, precision=0.0001, verbose=True)
                _ = test_final_policy(env=env, policy=policy, verbose=True)
        else:
            assert solver == 'Q-Learning', 'Check solver name.'
            for alpha_decay in [0.9, 0.6, 0.4, 'harmonic']:
                hyperparam_study(alpha_decay=alpha_decay, param='alpha decay', val=alpha_decay)
            for epsilon in [0.9, 0.6, 0.4, 0.1]:
                hyperparam_study(start_epsilon=epsilon, param='epsilon', val=epsilon)
            for epsilon_decay in [0.9, 0.6, 0.4, 0.1]:
                hyperparam_study(epsilon_decay=epsilon_decay, param='epsilon decay', val=epsilon_decay)
            for alpha in [0.9, 0.6, 0.4, 0.1]:
                hyperparam_study(base_alpha=alpha, param='alpha', val=alpha)
            for gamma in [0.8, 0.9, 0.95, 0.99]:
                hyperparam_study(gamma=gamma, param='gamma', val=gamma)

        t0 = perf_counter()
        if size == 8:
            policy, stats, values = Q_learning(env=env, gamma=0.99, base_alpha=0.6, alpha_decay=0.6, alpha_decay_speed=100, \
                    start_epsilon=0.3, epsilon_decay=0.4, epsilon_floor=0.01, num_episodes=200000, tol=0.00001, verbose=True)
            
            to_print = np.reshape(np.array(policy),(8,8))
        
        else:
            assert size == 32
            policy, stats, values = Q_learning(env=env, gamma=0.999, base_alpha=0.9, alpha_decay=0.99, alpha_decay_speed=100, \
                    start_epsilon=0.3, epsilon_decay=0.1, epsilon_floor=0.01, num_episodes=200000, tol=0.00001, verbose=True)

            to_print = np.reshape(np.array(policy),(32,32))[-8:,-8:]
        
        print(to_print)

        print(f'Finished {solver} in {round((perf_counter()-t0)/60,2)} minutes.')
        print('Now testing final policy.')
        _, _ = test_final_policy(env=env, policy=policy)
    

size_range = [(8, -0.001), (32, -0.002)]

for size, penalty in size_range:
    run_experiment(size=size, penalty=penalty)
