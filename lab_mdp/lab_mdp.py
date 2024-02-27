import numpy as np

# Setting env 
n_states = 16
n_actions = 4
goal_state = 15
obstacle_states = [1,7,9,11,13]

# Init Q-table
q_table = np.zeros((n_states,n_actions))

# Setting Q-learning 
learning_rate = 0.1
discount_rate = 0.99
episodes = 100
max_step_per_episode = 99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.01

def state_to_position(state):
    return (state // 4, state % 4)

def position_to_state(position):
    return position[0] * 4 + position[1]

def is_valid_position(position):
    state = position_to_state(position)
    if state in obstacle_states:
        return False
    return True

def simulate_action(state,action):
    x,y = state_to_position(state)
    if action == 0 :
        new_x , new_y = max(x-1,0),y
    elif action == 1 :
        new_x , new_y = min(x+1,3),y
    elif action == 2 :
        new_x , new_y = x , max(y-1,0)
    elif action == 3 :
        new_x , new_y = x , min(y+1,3)
    else : 
        new_x , new_y = x,y

    if not is_valid_position((new_x,new_y)):
        return state,-0.1
    
    new_state = position_to_state((new_x,new_y))
    reward = 10 if new_state == goal_state else -0.01
    return new_state,reward

def print_grind_and_position(state):
    grid = np.full((4,4), ' ')
    start_position = (0,0)
    goal_position = state_to_position(goal_state)
    for obstacle_state in obstacle_states:
        obstacle_position = state_to_position(obstacle_state)
        grid[obstacle_position] = 'X'
    agent_position = state_to_position(state)

    grid[start_position] = 'S'
    grid[goal_position] = 'G'
    if agent_position not in obstacle_states :
        grid[agent_position] = '0'

    print("\nGrid :")
    for row in grid : 
        print(' '. join(row))

for episode in range(episodes) : 
    state = 0 
    done = False 

    for step in range(max_step_per_episode) :
        if np.random.rand() < epsilon : 
            action = np.random.randint(0,n_actions)
        else : 
            action = np.argmax(q_table[state, :])
        
        new_state , reward = simulate_action(state,action)

        q_table[state , action] = q_table[state,action] + learning_rate * (
            reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])
        
        state = new_state

        if episode == episodes -1 : 
            print(f"Step {step+1} : State = {state}, Action = {action}, Reward = {reward}")
            print_grind_and_position(state)

        if state == goal_state :    
            break 

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

print("\nFinal Q-Table Values")
print(q_table)

