# import main_mfrl_REINFORCE_vanilla as Vanilla
# import main_mfrl_REINFORCE_recurrent as Recurrent
# import main_mfrl_REINFORCE_RA2C as RA2C
# import main_mfrl_REINFORCE_A2C as A2C
import main_mfrl_ra2c as RA2C
import main_mfrl_ra2cfull as RA2Cfull
import main_mfrl_a2c as A2C
import main_mfrl_reinforce as Reinforce
import main_mfrl_reinforcefull as Reinforcefull


from mfrl_lib.lib import *

import torch
from torch.distributions import Categorical
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_model(filepath, device):
    return torch.load(filepath, map_location=device)

def averaging_models(model_files, device='cpu'):
    # Load the first model to get the structure
    base_model = load_model(model_files[0], device)
    
    # Initialize a dict to store the sum of parameters
    averaged_params = {}
    for name, param in base_model.items():
        averaged_params[name] = torch.zeros_like(param)

    # Sum up the parameters from all models
    for file in model_files:
        model = load_model(file, device)
        for name, param in model.items():
            averaged_params[name] += param

    # Divide by the number of models to get the average
    for name in averaged_params:
        averaged_params[name] /= len(model_files)

    return averaged_params

def fuse_ra2c_models(model_pattern, output_file, device='cpu'):
    # Get all model files matching the pattern
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise ValueError(f"No model files found matching the pattern: {model_pattern}")

    print(f"Found {len(model_files)} model files.")

    # Average the models
    averaged_model = averaging_models(model_files, device)

    # Save the averaged model
    torch.save(averaged_model, output_file)
    print(f"Fused model saved to {output_file}")

# if GPU is to be used
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

topology.show_adjacency_matrix()

def test_model(simmode=None, max_episodes=20, max_steps=300):
    # Create the agents
    if simmode == "RA2C" or simmode == "RA2Cfed":
        agents = [RA2C.Agent(topology, i, arrival_rate[i]) for i in range(node_n)]
    elif simmode == "RA2Cfull":
        agent = RA2Cfull.Agent(topology, n_obs=4*node_n+1, n_act=node_n, arrival_rate=arrival_rate)
    elif simmode == "A2C":
        agents = [A2C.Agent(topology, i, arrival_rate[i]) for i in range(node_n)]
    elif simmode == "reinforce" or simmode == "fixedprob":
        agents = [Reinforce.Agent(topology, i) for i in range(node_n)]
    elif simmode == "reinforcefull":
        agent = Reinforcefull.Agent(topology, n_obs=2*node_n+1, n_act=2**node_n)
    else:
        raise ValueError("Invalid simmode.")
    
    # Load the trained models
    if simmode == "RA2C":
        for i in range(topology.n):
            agents[i].pinet.load_state_dict(torch.load(f"models/RA2C_{topo_string}_agent_{i}_{timestamp}.pth", map_location=device))
    elif simmode == "RA2Cfull":
        agent.pinet.load_state_dict(torch.load(f"models/RA2Cfull_{topo_string}_{timestamp}.pth", map_location=device))
    elif simmode == "RA2Cfed":
        model_pattern = f"models/RA2C_{topo_string}_agent_*_{timestamp}.pth"
        output_file = f"models/RA2Cfed_{topo_string}_{timestamp}.pth"
        fuse_ra2c_models(model_pattern, output_file, device)
        for i in range(topology.n):
            agents[i].pinet.load_state_dict(torch.load(output_file, map_location=device))
    elif simmode == "A2C":
        for i in range(topology.n):
            agents[i].pinet.load_state_dict(torch.load(f"models/A2C_{topo_string}_agent_{i}_{timestamp}.pth", map_location=device))
    elif simmode == "reinforce" or simmode == "fixedprob":
        for i in range(topology.n):
            agents[i].pinet.load_state_dict(torch.load(f"models/reinforce_{topo_string}_agent_{i}_{timestamp}.pth", map_location=device))
    elif simmode == "reinforcefull":
        agent.pinet.load_state_dict(torch.load(f"models/reinforcefull_{topo_string}_{timestamp}.pth", map_location=device))
        
    
    total_reward = 0
    df = pd.DataFrame()
    for n_epi in tqdm(range(max_episodes)):
        if simmode == "RA2Cfull" or simmode == "reinforcefull":
            obs = agent.env.reset()[0]
            # states = torch.from_numpy(agent.env.reset()[0].astype('float32')).unsqueeze(0).to(device)
            states = agent.flatten_observation(obs)
            states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            probs = None
            reward_per_epi = 0
            h = torch.zeros(1, 1, 32).to(device)
            c = torch.zeros(1, 1, 32).to(device)
        else:
            states = [torch.from_numpy(agent.env.reset()[0].astype('float32')).unsqueeze(0).to(device) for agent in agents]
            probs = [None]*node_n
            reward_per_epi = 0
            h = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
            c = [torch.zeros(1, 1, 32).to(device) for _ in range(node_n)]
        
        for t in range(max_steps):
            # Calculate the action for each agent
            actions = []
            next_states = []
            aoi_all = []
            if simmode == "RA2Cfull" or simmode == "reinforcefull":
                with torch.no_grad():
                    if simmode == "RA2Cfull":
                        prob, a, h, c, _, _, _ = agent.pinet.sample_action(states, h, c)
                    elif simmode == "reinforcefull":
                        prob = agent.pinet.sample_action(states)
                    # m = Categorical(prob)
                    # action = m.sample().item()
                    # action = a.item()
                    # actions.append(action)
                    actions = np.array(a.cpu().int())
                    
                next_state, reward_inst, _, _, _ = agent.env.step(actions)
                # next_state = torch.from_numpy(next_state.astype('float32')).unsqueeze(0).to(device)
                next_state = agent.flatten_observation(next_state)
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
                aoi_all.append(agent.env.age)
                reward_per_epi += reward_inst
                states = next_state
                df_index = pd.DataFrame(data=[[n_epi, t]], columns=['episode', 'epoch'])
                df_aoi = pd.DataFrame(data=aoi_all, columns=[f'aoi_{node}' for node in range(node_n)])
                df_action = pd.DataFrame(data=[actions], columns=[f'action_{node}' for node in range(node_n)])
                df_reward = pd.DataFrame(data=[[reward_per_epi]], columns=['reward'])
                df1 = pd.concat([df_index, df_aoi, df_action, df_reward], axis=1)
                df = pd.concat([df, df1])
            else:
                for i in range(node_n):
                    with torch.no_grad():
                        if simmode == "RA2C" or simmode == "RA2Cfed":
                            probs[i], a, h[i], c[i], _, _, _ = agents[i].pinet.sample_action(states[i], h[i], c[i])
                        elif simmode == "A2C":
                            probs[i], a, _, _, _ = agents[i].pinet.sample_action(states[i])
                        elif simmode == "recurrent":
                            probs[i], h[i], c[i], _, _ = agents[i].pinet.sample_action(states[i], h[i], c[i])
                        elif simmode == "reinforce":
                            probs[i] = agents[i].pinet.sample_action(states[i])
                        elif simmode == "fixedprob":
                            num_adjacent = sum(topology.adjacency_matrix[i])
                            txprob = 1/(num_adjacent+1)
                            probs[i] = torch.tensor([[[1-txprob, txprob]]]).to(device)
                        # action = Categorical(probs[i]).sample().item()
                        action = a.item()
                        actions.append(action)
                for i in range(node_n):
                    agents[i].env.set_all_actions(actions)
                    next_state, reward_inst, _, _, _ = agents[i].env.step(actions[i])
                    next_states.append(torch.from_numpy(next_state.astype('float32')).unsqueeze(0).to(device))
                    aoi_all.append(agents[i].env.age)
                    reward_per_epi += reward_inst
                    
                states = next_states
                df_index = pd.DataFrame(data=[[n_epi, t]], columns=['episode', 'epoch'])
                df_aoi = pd.DataFrame(data=[aoi_all], columns=[f'aoi_{node}' for node in range(node_n)])
                df_action = pd.DataFrame(data=[actions], columns=[f'action_{node}' for node in range(node_n)])
                df_reward = pd.DataFrame(data=[[reward_per_epi/node_n]], columns=['reward'])
                df1 = pd.concat([df_index, df_aoi, df_action, df_reward], axis=1)
                df = pd.concat([df, df1])
        total_reward += reward_per_epi
    average_reward = total_reward/(max_episodes*node_n)
    print(f"Average reward for {simmode}: {average_reward:.4f}")
    return df, average_reward

timestamp = FIXED_TIMESTAMP
log_folder = "test_logs"
os.makedirs(log_folder, exist_ok=True)

# for mode in ["RA2C", "RA2Cfed", "RA2Cfull", "A2C", "reinforce", "reinforcefull", "fixedprob"]:
for mode in ["RA2C", "RA2Cfull", "A2C"]:
# for mode in ["RA2Cfull"]:
    print(f"Testing model for {mode}...")
    MAX_EPISODES = 10
    MAX_STEPS = 200
    df, avg_reward = test_model(simmode=mode, max_episodes=MAX_EPISODES, max_steps=MAX_STEPS)
    
    # Define the full path for the CSV file
    filename = f"test_log_{mode}_{topo_string}_{timestamp}.csv"
    file_path = os.path.join(log_folder, filename)
    
    # Save the DataFrame to the test_log subfolder
    df.to_csv(file_path, index=False)
    
    print(f"Results for {mode} saved to {file_path}")
    print(f"Final average reward for {mode}: {avg_reward:.4f}\n")