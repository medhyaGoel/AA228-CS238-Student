import csv
import time

def lookahead(mdpTransitions, mdpRewards, s, a, discount_factor, total_states, U):
    value = mdpRewards[(s, a)] if (s, a) in mdpRewards else 0.0
    
    for next_state in mdpTransitions.get((s, a), {}):
        prob = mdpTransitions[(s, a)][next_state]
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
            reward = int(line[2])
            next_state = int(line[3])
            if (state, action) not in mdpCounts:
                mdpCounts[(state, action)] = {}
                mdpRewards[(state, action)] = 0.0
            if next_state not in mdpCounts[(state, action)]:
                mdpCounts[(state, action)][next_state] = 0
            mdpCounts[(state, action)][next_state] += 1
            mdpRewards[(state, action)] += reward
    return mdpCounts, mdpRewards


def modelEstimator(mdpCounts, mdpRewards, total_states):
    mdpTransitions = {} # {(state, action) -> {next_state: probability}}
    mdpRewardsAvg = {} # {(state, action) -> average_reward}
    for (state, action) in mdpCounts.keys():
        total = sum(mdpCounts[(state, action)].values())
        if total > 0:
            mdpTransitions[(state, action)] = {next_state: count / total for next_state, count in mdpCounts[(state, action)].items()}
            mdpRewardsAvg[(state, action)] = mdpRewards[(state, action)] / total
    return mdpTransitions, mdpRewardsAvg

def valueIteration(mdpTransitions, mdpRewards, discount_factor, total_states, total_actions, theta=1e-6):
    U = [0.0 for _ in range(total_states)]
    policy = [0 for _ in range(total_states)]
    while True:
        delta = 0
        for s in range(1, total_states + 1):
            best_a = backup(mdpTransitions, mdpRewards, s, discount_factor, total_states, total_actions, U)
            new_val = lookahead(mdpTransitions, mdpRewards, s, best_a, discount_factor, total_states, U)
            delta = max(delta, abs(new_val - U[s-1]))
            U[s-1] = new_val
            policy[s-1] = best_a
        if delta < theta:
            break
    return policy

def solve_policy(discount_factor, total_states, total_actions, file):
    start_time = time.time()
    mdpCounts, mdpRewards = mdpCreator(f"data/{file}.csv")
    mdpTransitions, mdpRewardsAvg = modelEstimator(mdpCounts, mdpRewards, total_states)
    policy = valueIteration(mdpTransitions, mdpRewardsAvg, discount_factor, total_states, total_actions)
    with open(f"{file}.policy", "w") as f:
        for state in range(1, total_states + 1):
            f.write(str(policy[state - 1]) + "\n")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


# solve_policy(0.95, 100, 4, "small") # small
# solve_policy(1, 50000, 7, "medium") # medium
solve_policy(0.95, 302020, 9, "large") # medium