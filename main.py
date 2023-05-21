import os
import sys
import sumo_rl

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

import sumo_rl


# def obs_fun_wave_wait():

if __name__ == "__main__":
    env = sumo_rl.parallel_env(
        net_file="sumo_files/4x4_grid_network.net.xml",
        route_file="sumo_files/4x4_grid_routes.rou.xml",
        # net_file="sumo_files/chry-test.net.xml",
        # route_file="sumo_files/chry-trips.trips.xml",
        out_csv_name="outputs/ppo",
        single_agent=False,
        use_gui=True,
        num_seconds=5000000,
    )
    env.reset(seed=33)
    s=False
    while env.agents:
    # this is where you would insert your policy
        a=env.agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  
        action_spaces=env.action_spaces
        a_s={agent: env.action_space(agent) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()