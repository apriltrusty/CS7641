from time import perf_counter
import gym
from gym_classics.dynamic_programming import policy_iteration, value_iteration
import numpy as np
import pandas as pd


def epsilon_greedy(env, Q, state, epsilon):
    if np.random.rand() < epsilon: action = env.action_space.sample()
    else: action = np.argmax(Q[state])
    return action


def Q_learning(env, gamma=0.05, start_epsilon=0.9, epsilon_decay=0.9, epsilon_floor=0.01, epsilon_decay_speed = 100, \
                alpha_decay='harmonic', alpha_decay_speed = 100, base_alpha=0.3, tol=0.0001, num_episodes=10000, verbose=False):
    
    stats = pd.DataFrame(columns=['size','episode','alpha','epsilon','alpha decay','epsilon decay', \
        'gamma','score','time','num steps','max value change', 'policy'])
    this_episode_stats = {}

    # Hyperparameters for Q-Learning
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
            total_reward += reward

            # Q-Learning update:
            # Q(s,a) <-- Q(s,a) + α * (r + γ max_a' Q(s',a') - Q(s,a))
            target = reward - Q[state, action]
            if not done:
                target += gamma * np.max(Q[next_state])
            
            Q[state, action] = Q[state, action] + alpha * target
            state = next_state

        max_value_change = np.max(np.abs(Q - old_Q))
        this_episode_stats['score'] = total_reward
        this_episode_stats['time'] = perf_counter() - t0
        this_episode_stats['num steps'] = num_steps
        this_episode_stats['max value change'] = max_value_change
        this_episode_stats['average Q'] = np.average(Q)
        this_episode_stats['policy'] = str(np.argmax(Q,axis=1))

        stats = stats.append(pd.DataFrame(this_episode_stats, index=[episode]))

        if verbose and episode % 100 == 0:
            print(f'Episode #{episode}\tReward:\t{round(total_reward,0)}\tEpsilon:\t{round(epsilon,3)}\tAlpha:\t{round(alpha,3)}\tValue Change:\t{round(max_value_change,4)}\t\tAvg Q:\t{np.average(Q)}')

        # check for convergence
        if episode > 1000:
            if max(stats['max value change'][-10:]) < tol:
                print('Q-Learning converged at episode {}. Time taken: {} minutes.'.format(episode, round(np.sum(stats['time'])/60,2)))
                converged = True
                break
    
    if not converged: print('Q-Learning did not converge in {} episodes. Time taken: {} minutes.'.format(episode, round(np.sum(stats['time'])/60,2)))
    final_policy = np.argmax(Q,axis=1)
    values = pd.DataFrame(data=Q)

    return final_policy, stats, values


def test_final_policy(env, policy=None, num_episodes=100, verbose=False):
    reward_log = []

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
            total_reward += reward
            state = next_state

        reward_log.append(total_reward)
        if verbose: print(total_reward)

    t1 = perf_counter()
    print('Ran {} episodes: Average reward: {} (+/-{}). Time: {} seconds.'.format(num_episodes, 
                                                    round(np.average(reward_log),2),
                                                    round(np.std(reward_log),2),
                                                    round(t1-t0,2)))
    
    return reward_log


def run_experiment(size=11, max_move=3):
    np.random.seed(0)

    if size == 32:
        env = gym.make('JacksCarRental-v0', size=size, max_move=max_move,
                    lot1_requests_distr=5, lot1_dropoffs_distr=13,
                    lot2_requests_distr=13, lot2_dropoffs_distr=5)
    
    else:
        assert size == 11
        env = gym.make('JacksCarRental-v0', size=size, max_move=max_move)

    env.seed(0)
    solvers = ['Policy Iteration','Value Iteration','Q-Learning']

    def hyperparam_study(gamma=0.1, base_alpha=0.9, alpha_decay=0.9, start_epsilon=0.9, epsilon_decay=0.9, param=None, val=None):
        print(f'\nResults for {param}={val}')
        t0 = perf_counter()
        policy, stats, values = Q_learning(env=env, gamma=gamma, base_alpha=base_alpha, alpha_decay=alpha_decay, alpha_decay_speed=100, \
            start_epsilon=start_epsilon, epsilon_decay=epsilon_decay, epsilon_floor=0.1, num_episodes=50000, tol=0.00001, verbose=True)

        print(f'Finished {solver} in {round(perf_counter()-t0,2)} seconds.')
        print('Now testing final policy.')
        _ = test_final_policy(env=env, policy=policy, verbose=True)

        assert solver == 'Q-Learning', 'Check solver, only Q-Learning accepted'
        with pd.ExcelWriter(f'Excel/jacks_{size}.xlsx',mode='a') as writer:
                    stats.to_excel(writer, sheet_name=f'Stats, {param}={val}')
                    values.to_excel(writer, sheet_name=f'Values, {param}={val}')

    for solver in solvers:
        print(f'======={solver} on size {size}=======')
        if solver == 'Policy Iteration':
            for gamma in [0.8, 0.9, 0.95, 0.99]:
                print(f'gamma={gamma}')
                policy = policy_iteration(env=env, discount=gamma, precision=0.0001, verbose=True)
                _ = test_final_policy(env=env, policy=policy, verbose=True)
        elif solver == 'Value Iteration':
            for gamma in [0.8, 0.9, 0.95, 0.99]:
                print(f'gamma={gamma}')
                policy = value_iteration(env=env, discount=gamma, precision=0.0001, verbose=True)
                _ = test_final_policy(env=env, policy=policy, verbose=True)
        else:
            assert solver == 'Q-Learning', 'Check solver name.'
            for alpha in [0.9, 0.6, 0.4, 0.1]:
                param, val='alpha', alpha
                hyperparam_study(base_alpha=alpha, param=param, val=val)
            for gamma in [0.9, 0.7, 0.3, 0.1]:
                param, val='gamma', gamma
                hyperparam_study(gamma=gamma, param=param, val=val)
            for start_epsilon in [0.9, 0.6, 0.4, 0.1]:
                param, val='start epsilon', start_epsilon
                hyperparam_study(start_epsilon=start_epsilon, param=param, val=val)
            for epsilon_decay in [0.9, 0.6, 0.4, 0.1]:
                param, val='epsilon decay', epsilon_decay
                hyperparam_study(epsilon_decay=epsilon_decay, param=param, val=val)
            for alpha_decay in [0.9, 0.6, 0.4, 'harmonic']:
                param, val='alpha decay', alpha_decay
                hyperparam_study(alpha_decay=alpha_decay,param=param, val=val)
            
            t0 = perf_counter()
            if size == 11:
                policy, stats, values = Q_learning(env=env, gamma=0.3, base_alpha=0.9, alpha_decay=0.6, alpha_decay_speed=100, \
                    start_epsilon=0.9, epsilon_decay=0.9, epsilon_floor=0.1, num_episodes=50000, tol=0.00001, verbose=True)
                
                to_print = np.reshape(np.array(policy),(11,11))
            
            else:
                assert size == 32, 'Check size of problem'
                policy, stats, values = Q_learning(env=env, gamma=0.1, base_alpha=0.9, alpha_decay=0.9, alpha_decay_speed=100, \
                    start_epsilon=0.99, epsilon_decay=0.99, epsilon_floor=0.1, num_episodes=50000, tol=0.00001, verbose=True)

                to_print = np.reshape(np.array(policy),(32,32))[-11:,-11:]

            print(to_print)
            
            print(f'Finished {solver} in {perf_counter()-t0} seconds.')
            print('Now testing final policy.')
            _ = test_final_policy(env=env, policy=policy)


size_range = [(11,3), (32,8)]

for size, max_move in size_range:
    run_experiment(size=size, max_move=max_move)
