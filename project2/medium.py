import csv
import time
import statistics

# Environment discretization parameters
low_pos, high_pos = -1.2, 0.6
low_vel, high_vel = -0.07, 0.07
num_pos_bins = 500
num_vel_bins = 100

def lookahead(mdpTransitions, mdpRewards, s, a, discount_factor, total_states, U):
    value = mdpRewards.get((s, a), -1.0)
    for next_state, prob in mdpTransitions.get((s, a), {}).items():
        value += discount_factor * prob * U[next_state - 1]
    return value

def backup(mdpTransitions, mdpRewards, s, discount_factor, total_states, total_actions, U):
    best_a = max(
        range(1, total_actions + 1),
        key=lambda a: lookahead(mdpTransitions, mdpRewards, s, a, discount_factor, total_states, U)
    )
    return best_a

def mdpCreator(filename):
    mdpCounts = {} # {(state, action) -> {next_state: count}}
    mdpRewards = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            state = int(line[0])
            action = int(line[1])
            reward = float(line[2])
            next_state = int(line[3])
            if (state, action) not in mdpCounts:
                mdpCounts[(state, action)] = {}
                mdpRewards[(state, action)] = 0.0
            mdpCounts[(state, action)][next_state] = mdpCounts[(state, action)].get(next_state, 0) + 1
            mdpRewards[(state, action)] += reward
    return mdpCounts, mdpRewards

def modelEstimator(mdpCounts, mdpRewards, total_states, estimated_goal_pos=None):
    mdpTransitions = {}  # {(state, action) -> {next_state: probability}}
    mdpRewardsAvg = {}   # {(state, action) -> average_reward}
    alpha = 0.01
    for (state, action), next_counts in mdpCounts.items():
        total = sum(next_counts.values()) + alpha * len(next_counts)
        mdpTransitions[(state, action)] = {next_state: (count + alpha) / total
                                           for next_state, count in next_counts.items()}
        mdpRewardsAvg[(state, action)] = mdpRewards[(state, action)] / sum(next_counts.values())
        
        # --- Reward shaping using estimated goal ---
        if estimated_goal_pos is not None:
            # Convert discrete state to environment position
            s0 = state - 1
            pos_bin = s0 % num_pos_bins
            pos_env = low_pos + pos_bin * (high_pos - low_pos) / (num_pos_bins - 1)
            distance_to_goal = max(0, estimated_goal_pos - pos_env)
            shaped_reward = 0.01 * distance_to_goal  # small reward for moving right
            mdpRewardsAvg[(state, action)] += shaped_reward
            
        # Clip rewards to avoid runaway values
        mdpRewardsAvg[(state, action)] = min(mdpRewardsAvg[(state, action)], 100.0)
        
    return mdpTransitions, mdpRewardsAvg

def policyIteration(mdpTransitions, mdpRewards, discount_factor=0.99, total_states=50000, total_actions=7, theta=1e-2, estimated_goal_pos=None):
    # --- Initialize value function with higher values near the goal ---
    U = [0.0] * total_states
    if estimated_goal_pos is not None:
        for s in range(1, total_states + 1):
            s0 = s - 1
            pos_bin = s0 % num_pos_bins
            pos_env = low_pos + pos_bin * (high_pos - low_pos) / (num_pos_bins - 1)
            U[s-1] = max(0, 100 * (pos_env - low_pos) / (estimated_goal_pos - low_pos))
    
    policy = [1] * total_states
    max_iterations = 500

    for it in range(max_iterations):
        delta = 0
        for s in range(1, total_states + 1):
            best_a = backup(mdpTransitions, mdpRewards, s, discount_factor, total_states, total_actions, U)
            new_val = lookahead(mdpTransitions, mdpRewards, s, best_a, discount_factor, total_states, U)
            delta = max(delta, abs(new_val - U[s-1]))
            U[s-1] = 0.5 * U[s-1] + 0.5 * new_val
            policy[s-1] = best_a
        print(f"Iteration {it}, delta = {delta}")
        if delta < theta:
            break
    return policy

def estimate_goal(filename, reward_threshold=50000):
    goal_states = set()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            state = int(line[0])
            reward = float(line[2])
            if reward >= reward_threshold:
                goal_states.add(state)
    # Convert discrete state to positions
    goal_positions_env = []
    for state in goal_states:
        s0 = state - 1
        pos_bin = s0 % num_pos_bins
        pos_env = low_pos + pos_bin * (high_pos - low_pos) / (num_pos_bins - 1)
        goal_positions_env.append(pos_env)
    return max(goal_positions_env) if goal_positions_env else high_pos

def solve_policy(discount_factor, total_states, total_actions, file):
    start_time = time.time()
    
    # Step 1: Estimate goal position from dataset
    estimated_goal_pos = estimate_goal(f"data/{file}.csv")
    print(f"Estimated goal position: {estimated_goal_pos:.4f}")
    
    # Step 2: Create MDP counts and rewards
    mdpCounts, mdpRewards = mdpCreator(f"data/{file}.csv")
    
    # Step 3: Estimate MDP model with reward shaping
    mdpTransitions, mdpRewardsAvg = modelEstimator(mdpCounts, mdpRewards, total_states, estimated_goal_pos)
    
    # Step 4: Policy iteration with initialized value function
    policy = policyIteration(mdpTransitions, mdpRewardsAvg, discount_factor, total_states, total_actions, estimated_goal_pos=estimated_goal_pos)
    
    # Step 5: Save policy
    with open(f"{file}.policy", "w") as f:
        for state in range(1, total_states + 1):
            f.write(str(policy[state - 1]) + "\n")
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

# Example usage:
solve_policy(1, 50000, 7, "medium")
