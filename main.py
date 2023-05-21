import os
import sys
import sumo_rl
import torch
import torch.nn as nn
from torch.autograd import Variable

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

import sumo_rl

class ActorModel(nn.Module):
    def __init__(self, wave_n, wait_n, neighbour_s_n,phases_n):
        super(ActorModel, self).__init__()
        self.wave_n=wave_n
        self.wait_n=wait_n
        self.neighbour_s_n=neighbour_s_n
        # Define the three FC layers with input sizes wave_n, wait_n, and neighbour_s_n
        self.fc1 = nn.Linear(wave_n, 128)
        self.fc2 = nn.Linear(wait_n, 32)
        self.fc3 = nn.Linear(neighbour_s_n, 64)
        # unbatched
        '''
        h_0: tensor of shape (D∗num_layers,Hout)(D∗num_layers,Hout​) for unbatched input or (D∗num_layers,N,Hout)(D∗num_layers,N,Hout​) containing the initial hidden state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.

        c_0: tensor of shape (D∗num_layers,Hcell)(D∗num_layers,Hcell​) for unbatched input or (D∗num_layers,N,Hcell)(D∗num_layers,N,Hcell​) containing the initial cell state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.
        '''
        self.hn = Variable(torch.zeros(1*1,548))
        self.cn = Variable(torch.zeros(1*1,548))
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=224, hidden_size=548, num_layers=1)
        self.linear=nn.Linear(548,phases_n)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, wave,wait,neighbour_s):
        # Pass input through the three FC layers
        x1 = self.fc1(wave)
        x2 = self.fc2(wait)
        x3 = self.fc3(neighbour_s)
        
        # Combine the outputs of the FC layers
        combined = torch.cat((x1, x2, x3), dim=1)
        
        # Prepare input for LSTM layer
        '''
        input: tensor of shape (L,Hin)(L,Hin​) for unbatched input, (L,N,Hin)(L,N,Hin​) when batch_first=False or (N,L,Hin)(N,L,Hin​) when batch_first=True containing the features of the input sequence. 
        where:
        N=batch sizeL=sequence lengthD=2 if bidirectional=True otherwise 1Hin=input_sizeHcell=hidden_sizeHout=proj_size if proj_size>0 otherwise hidden_size
        N=L=D=Hin​=Hcell​=Hout​=​batch sizesequence length2 if bidirectional=True otherwise 1input_sizehidden_sizeproj_size if proj_size>0 otherwise hidden_size​
        '''
        lstm_input = combined.view(1, 224)
        
        # Initialize hidden state and cell state for LSTM layer
        
        
        # Pass input through the LSTM layer
        lstm_output, (hn, cn) = self.lstm(lstm_input, (hn, cn))

        '''
        output: tensor of shape (L,D∗Hout)(L,D∗Hout​) for unbatched input, (L,N,D∗Hout)(L,N,D∗Hout​) when batch_first=False or (N,L,D∗Hout)(N,L,D∗Hout​) when batch_first=True containing the output features (h_t) from the last layer of the LSTM, for each t
        '''
        lin_outp=self.linear(lstm_output[:,:])
        softmax_output=self.softmax(lin_outp)
        return softmax_output

class CriticModel(nn.Module):
    def __init__(self, wave_n, wait_n, neighbour_s_n):
        super(CriticModel, self).__init__()
        self.wave_n=wave_n
        self.wait_n=wait_n
        self.neighbour_s_n=neighbour_s_n
        # Define the three FC layers with input sizes wave_n, wait_n, and neighbour_s_n
        self.fc1 = nn.Linear(wave_n, 128)
        self.fc2 = nn.Linear(wait_n, 32)
        self.fc3 = nn.Linear(neighbour_s_n, 64)
        # unbatched
        '''
        h_0: tensor of shape (D∗num_layers,Hout)(D∗num_layers,Hout​) for unbatched input or (D∗num_layers,N,Hout)(D∗num_layers,N,Hout​) containing the initial hidden state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.

        c_0: tensor of shape (D∗num_layers,Hcell)(D∗num_layers,Hcell​) for unbatched input or (D∗num_layers,N,Hcell)(D∗num_layers,N,Hcell​) containing the initial cell state for each element in the input sequence. Defaults to zeros if (h_0, c_0) is not provided.
        '''
        self.hn = Variable(torch.zeros(1*1,548))
        self.cn = Variable(torch.zeros(1*1,548))
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=224, hidden_size=548, num_layers=1)
        self.linear=nn.Linear(548,1)
    def forward(self, wave,wait,neighbour_s):
        # Pass input through the three FC layers
        x1 = self.fc1(wave)
        x2 = self.fc2(wait)
        x3 = self.fc3(neighbour_s)
        
        # Combine the outputs of the FC layers
        combined = torch.cat((x1, x2, x3), dim=1)
        
        # Prepare input for LSTM layer
        '''
        input: tensor of shape (L,Hin)(L,Hin​) for unbatched input, (L,N,Hin)(L,N,Hin​) when batch_first=False or (N,L,Hin)(N,L,Hin​) when batch_first=True containing the features of the input sequence. 
        where:
        N=batch sizeL=sequence lengthD=2 if bidirectional=True otherwise 1Hin=input_sizeHcell=hidden_sizeHout=proj_size if proj_size>0 otherwise hidden_size
        N=L=D=Hin​=Hcell​=Hout​=​batch sizesequence length2 if bidirectional=True otherwise 1input_sizehidden_sizeproj_size if proj_size>0 otherwise hidden_size​
        '''
        lstm_input = combined.view(1, 224)
        
        # Initialize hidden state and cell state for LSTM layer
        
        
        # Pass input through the LSTM layer
        lstm_output, (hn, cn) = self.lstm(lstm_input, (hn, cn))

        '''
        output: tensor of shape (L,D∗Hout)(L,D∗Hout​) for unbatched input, (L,N,D∗Hout)(L,N,D∗Hout​) when batch_first=False or (N,L,D∗Hout)(N,L,D∗Hout​) when batch_first=True containing the output features (h_t) from the last layer of the LSTM, for each t
        '''
        lin_outp=self.linear(lstm_output[:,:])

        return lstm_output
# def obs_fun_wave_wait():

if __name__ == "__main__":
    env = sumo_rl.parallel_env(
        # net_file="sumo_files/4x4_grid_network.net.xml",
        # route_file="sumo_files/4x4_grid_routes.rou.xml",
        net_file="sumo_files/v1_4x4_grid.net.xml",
        route_file="sumo_files/v1_4x4_grid.rou.xml",
        # net_file="sumo_files/chry-test.net.xml",
        # route_file="sumo_files/chry-trips.trips.xml",
        out_csv_name="outputs/",
        single_agent=False,
        use_gui=True,
        # begin_time=10,
        num_seconds=5000000,
        min_green=5,
        max_green=60,
        delta_time=5
    )
    observation=env.reset(seed=33)
    while agents:=env.agents:
    # this is where you would insert your policy
        a_s={agent: env.action_space(agent) for agent in agents}
        actions = {agent: env.action_space(agent).sample() for agent in agents}  
        observations, rewards, terminations, truncations, infos = env.step(actions)
        pass
    env.close()