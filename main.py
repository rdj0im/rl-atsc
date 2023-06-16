import os
import sys
import sumo_rl
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import math
import traci
import numpy as np

# Set environment variable
os.environ['SUMO_HOME'] = '/usr/share/sumo'
os.environ['LIBSUMO_AS_TRACI'] = '1' #Optional: for a huge performance boost (~8x) with Libsumo (No GUI)

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
torch.autograd.set_detect_anomaly(True)
TIME_STEPS=1000000
GAMMA=0.99
ALPHA=0.75 # spacial discount base
A=0.5 #tradeoff coeff
lr_actor=5e-4
lr_critic=2.5e-4
Beta=0.01 #regularization coeff
batch_size=5

def q_w_reward_fn(self):
    queue=sum([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes
        ])
    wait=sum([
            self.sumo.lane.getWaitingTime(lane) for lane in self.lanes
        ])
    return -(queue+A*wait)

def junc_distance(fromJunction, toJunction):
    position1 = traci.junction.getPosition(fromJunction)
    position2 = traci.junction.getPosition(toJunction)
    x1, y1 = position1
    x2, y2 = position2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
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

    def reset_lstm():
        self.hn = Variable(torch.zeros(1*1,548))
        self.cn = Variable(torch.zeros(1*1,548))

    def forward(self, wave,wait,neighbour_s):
        # Pass input through the three FC layers
        x1 = self.fc1(wave)
        x2 = self.fc2(wait)
        x3 = self.fc3(neighbour_s)
        
        # Combine the outputs of the FC layers
        combined = torch.cat((x1, x2, x3), dim=0)
        
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
        lstm_output, (self.hn,self.cn) = self.lstm(lstm_input, (self.hn.detach(),self.cn.detach()))

        '''
        output: tensor of shape (L,D∗Hout)(L,D∗Hout​) for unbatched input, (L,N,D∗Hout)(L,N,D∗Hout​) when batch_first=False or (N,L,D∗Hout)(N,L,D∗Hout​) when batch_first=True containing the output features (h_t) from the last layer of the LSTM, for each t
        '''
        # lin_inp=lstm_output.clone()
        lin_outp=self.linear(lstm_output[:,:])
        softmax_output=self.softmax(lin_outp.clone())
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
        # '''

        # hidden_state = torch.zeros(1, x.size(0), 524).requires_grad_()
        # cell_state = torch.zeros(1, x.size(0), 524).requires_grad_()
        self.hn = Variable(torch.zeros(1*1,548))
        self.cn = Variable(torch.zeros(1*1,548))
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=224, hidden_size=548, num_layers=1)
        self.linear=nn.Linear(548,1)


    def reset_lstm():
        self.hn = Variable(torch.zeros(1*1,548))
        self.cn = Variable(torch.zeros(1*1,548))
        

    def forward(self, wave,wait,neighbour_s):
        # Pass input through the three FC layers
        x1 = self.fc1(wave)
        x2 = self.fc2(wait)
        x3 = self.fc3(neighbour_s)
        
        # Combine the outputs of the FC layers
        combined = torch.cat((x1, x2, x3), dim=0)
        
        # Prepare input for LSTM layer
        '''
        input: tensor of shape (L,Hin)(L,Hin​) for unbatched input, (L,N,Hin)(L,N,Hin​) when batch_first=False or (N,L,Hin)(N,L,Hin​) when batch_first=True containing the features of the input sequence. 
        where:
        N=batch sizeL=sequence lengthD=2 if bidirectional=True otherwise 1Hin=input_sizeHcell=hidden_sizeHout=proj_size if proj_size>0 otherwise hidden_size
        N=L=D=Hin​=Hcell​=Hout​=​batch sizesequence length2 if bidirectional=True otherwise 1input_sizehidden_sizeproj_size if proj_size>0 otherwise hidden_size​
        '''
        lstm_input = combined.view(1, 224)
        
        # Initialize hidden state and cell state for LSTM layer
        # h0 = Variable(torch.zeros(1*1,548))
        # c0 = Variable(torch.zeros(1*1,548))
        
 

        # Pass input through the LSTM layer
        # hn_t=self.hn.clone()
        # cn_t= self.cn.clone()
        lstm_output, (self.hn,self.cn) = self.lstm(lstm_input, (self.hn.detach(),self.cn.detach()))
        # self.hn=h0
        # self.cn=c0
        '''
        output: tensor of shape (L,D∗Hout)(L,D∗Hout​) for unbatched input, (L,N,D∗Hout)(L,N,D∗Hout​) when batch_first=False or (N,L,D∗Hout)(N,L,D∗Hout​) when batch_first=True containing the output features (h_t) from the last layer of the LSTM, for each t
        '''
        # lin_inp=lstm_output.clone()
        lin_outp=self.linear(lstm_output[:,:])

        return lin_outp
# def obs_fun_wave_wait():

class Environment:
    def get_env(self):
        os.environ['LIBSUMO_AS_TRACI'] = '1' 
        env = sumo_rl.parallel_env(
            # net_file="sumo_files/4x4_grid_network.net.xml",
            # route_file="sumo_files/4x4_grid_routes.rou.xml",
            net_file="sumo_files/v1_4x4_grid.net.xml",
            # net_file="sumo_files/v1_4x4_grid_tl_adj.net.xml",
            route_file="sumo_files/1Mtimesteps.rou.xml",
            # net_file="sumo_files/chry-test.net.xml",
            # route_file="sumo_files/chry-trips.trips.xml",
            out_csv_name="outputs/",
            single_agent=False,
            use_gui=False,
            # begin_time=10,
            num_seconds=5000000,
            min_green=5,
            max_green=60,
            delta_time=5,
            # begin_time=600,
            reward_fn=q_w_reward_fn
        )
        return env

class Networks:
    def __init__(self):
        self.networks={}
        self.optimizers={}
    def init_env_models(self,env,observations,neighbourhoods):
        for signal_id in env.agents:
            phases_n=env.action_space(signal_id).n
            wave_n=len(observations[signal_id]['wave'])
            wait_n=len(observations[signal_id]['wait'])
            neighbour_n=sum( len(observations[nghbr]['wait'])+len(observations[nghbr]['wait']) for nghbr in neighbourhoods[signal_id] )
            self.init_signal_model(signal_id, wave_n, wait_n, neighbour_n, phases_n)
    def init_signal_model(self,signal_id,wave_n,wait_n,neighbour_n,phases_n):
            actor_network=ActorModel(wave_n,wait_n,neighbour_n,phases_n)
            critic_network=CriticModel(wave_n,wait_n,neighbour_n)
            self.networks[signal_id]=(actor_network,critic_network)
            self.optimizers[signal_id]=(
                optim.RMSprop(actor_network.parameters(), lr=lr_actor),
                optim.RMSprop(critic_network.parameters(), lr=lr_critic)
                )

    def get_signal_actor(self,signal_id):
        try:
            return self.networks[signal_id][0]
        except KeyError as e:
            raise KeyError(f"Model for {signal_id} not defined")

    def get_signal_critic(self,signal_id):
        try:
            return self.networks[signal_id][1]
        except KeyError as e:
            raise KeyError(f"Model for {signal_id} not defined")

    def get_signal_actor_optim(self,signal_id):
        try:
            return self.optimizers[signal_id][0]
        except KeyError as e:
            raise KeyError(f"Model for {signal_id} not defined")

    def get_signal_critic_optim(self,signal_id):
        try:
            return self.optimizers[signal_id][1]
        except KeyError as e:
            raise KeyError(f"Model for {signal_id} not defined")


    def seq_end(self):
        for _,nets in self.networks.items():
            nets[0].reset_lstm()
            nets[1].reset_lstm()

def init_signal_batch_array_vals(signals):
    return { signal: np.empty(batch_size,dtype=np.float64)for signal in signals }

def init_signal_batch_array_dicts(signals):
    batch={signal: np.empty(batch_size,dtype=dict) for signal in signals}
    # batch=np.empty((batch_size,),dtype=dict)
    # batch[:]=[{} for _ in range(batch_size)]
    return batch

def get_neighbour_states(signal,observations):
    neighbour_states=[]
    for nbr in neighbourhoods[signal]:
        neighbour_states.extend(torch.tensor([ALPHA*x for x in observations[nbr]['wave']],dtype=torch.float32))
        neighbour_states.extend(torch.tensor([ALPHA*x for x in observations[nbr]['wait']],dtype=torch.float32))
    neighbour_s=torch.tensor(neighbour_states,dtype=torch.float32)
    return neighbour_s

def valueLoss(local_return,value):
    # value loss
    mse_loss = nn.MSELoss()
    mse = mse_loss(local_return,value)
    return 0.5 * mse

def policyLoss(action_probs,advantages):
    terms=torch.zeros(batch_size)
    for i in range(batch_size):
        action_index=torch.argmax(action_probs[i])
        sel_action=action_probs[i].flatten()[action_index]
        reg_term=Beta*torch.sum(torch.mul(action_probs[i], torch.log
        (action_probs[i])))
        terms[i]=torch.log(sel_action)*advantages[i]-reg_term

    loss=-1*torch.mean(terms)
    return loss

if __name__ == "__main__":

    #init env
    env=Environment().get_env()
    observations,info=env.reset()
    signals=env.agents
    # init arrays
    # assuming uniform neighbourhoods, lanes
    # every other signal is a neighbour
    neighbourhoods={signal: [item for item in signals if item!=signal] for signal in signals}
    # wait=torch.tensor([0]*wait_n,dtype=torch.float32)
    # wave=torch.tensor([0]*wave_n,dtype=torch.float32)
    # neighbour_states=torch.tensor([0]*neighbour_n,dtype=torch.float32)
    #init models
    networks=Networks()
    networks.init_env_models(env, observations, neighbourhoods)
    # batch=[{}]*batch_size
    # array of dicts
    # batch=init_batch_array()
    batch=init_signal_batch_array_dicts(signals)
    # dict indexed by signal; value - array (len batch size) where each elem corresponds to an exp in the batch
    estimated_return=init_signal_batch_array_vals(signals)
    probs={ signal:torch.zeros(batch_size) for signal in signals}
    all_action_probs={ signal:[0]*batch_size for signal in signals}
    # all_action_probs={}
    batch_ep_no=0
    for _ in range(TIME_STEPS):

        # once a batch is complete
        if batch_ep_no==batch_size: 
            for signal,returns in estimated_return.items():
                # initialize tensors
                local_returns=torch.zeros(batch_size)
                advantages=torch.zeros(batch_size)
                probs=torch.zeros(batch_size)
                log_probs=torch.zeros(batch_size)
                values=torch.zeros(batch_size)
                # get end state
                wave=torch.tensor(observations[signal]['wave'],dtype=torch.float32)
                wait=torch.tensor(observations[signal]['wait'],dtype=torch.float32)
                neighbour_s=get_neighbour_states(signal,observations)
                # get critic net
                critic=networks.get_signal_critic(signal)
                # end state value
                value_end_state=critic(wave,wait,neighbour_s)
                # compute values for each exp
                for i,exp_ret in enumerate(returns[:-1]) :
                    local_return=estimated_return[signal][i]+GAMMA**(batch_size-(i+1))*value_end_state
                    cur_state=batch[signal][i+1]
                    value_cur_state=critic(cur_state['wave'],cur_state['wait'],cur_state['neighbour_s'])
                    advantage=local_return-value_cur_state
                    local_returns[i]=local_return
                    advantages[i]=advantage
                    values[i]=value_cur_state
                local_returns[-1]=estimated_return[signal][-1]+value_end_state
                # GAMMA^0=0
                advantages[-1]=local_return[-1] # +value_end_state-value_end_state
            


            actor_model=networks.get_signal_actor(signal)
            _=actor_model(wave,wait,neighbour_s)
            act_probs_clone=[x.clone() for x in all_action_probs[signal]]
            policy_loss=policyLoss(act_probs_clone,advantages.clone())
            actor_optim=networks.get_signal_actor_optim(signal)
            actor_optim.zero_grad()
            policy_loss.backward(retain_graph=True)
            actor_optim.step()
            _=critic(wave,wait,neighbour_s)
            value_loss=valueLoss(local_returns.clone(), values.clone())
            critic_optim=networks.get_signal_critic_optim(signal)
            critic_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            critic_optim.step()

                


            # reset batch
            # assumes junctions remain the same with same names, so just overwrites current entries in coming iterations
            batch_ep_no=0

        actions={}
        for signal in signals:
            # model inputs
            wave=torch.tensor(observations[signal]['wave'],dtype=torch.float32)
            wait=torch.tensor(observations[signal]['wait'],dtype=torch.float32)
            neighbour_s=get_neighbour_states(signal,observations)
            
            # get model
            actor=networks.get_signal_actor(signal)
            # call model, get output
            action_probs=actor(wave,wait,neighbour_s)
            # store for updation 
            batch[signal][batch_ep_no]={'wave':wave,'wait':wait,'neighbour_s':neighbour_s}
            # get current action
            prob=torch.argmax(action_probs)
            all_action_probs[signal][batch_ep_no]=action_probs
            actions[signal]=prob.item()
        # step env based on actions obtained            
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # if done
        # episode end
        if any(terminations.values()) or any(truncations.values()):
            observations,info=env.reset()
            networks.seq_end()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        # compute global rewards for each signal for current batch
        glb_rew={}
        for signal in signals:
            glb_rew=sum(a:=(ALPHA**junc_distance(signal,peer))*rewards[peer] for peer in neighbourhoods[signal])
            tau=batch_ep_no
            estimated_return[signal][batch_ep_no]=0
            while tau>=0: #possible OBOE
                estimated_return[signal][tau]+=GAMMA**(batch_ep_no-tau)*glb_rew
                tau-=1
        # for signal,reward in rewards.items() : batch_rewards[batch_ep_no][signal]=reward
   
        # agents=env.agents
        # a_s={agent: env.action_space(agent) for agent in agents}
        # actions = {agent: env.action_space(agent).sample() for agent in agents}  
        # observations, rewards, terminations, truncations, infos = env.step(actions)
        batch_ep_no+=1
    env.close()