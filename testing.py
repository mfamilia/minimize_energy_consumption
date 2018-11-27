import os
import numpy as np
import random as rn
from environment import Environment
from keras.models import load_model

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

env = Environment(initial_number_users = 20, initial_rate_data = 30)
env.train = False
model = load_model("model.h5")

def enery_direction(action, boundary, step):
    action_change = action - boundary
    if (action_change < 0):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action_change) * step
    
    return energy_ai, direction

current_state, _, _ = env.observe()
month_minutes = 30 * 24 * 60
year_minutes = 12 * month_minutes 

for timestep in range(0, year_minutes):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    energy_ai, direction = enery_direction(action, direction_boundary, temperature_step)
    month = int(timestep / month_minutes) 
    next_state, _, _ = env.update_env(direction, energy_ai, month)
    current_state = next_state

print("\n")     
print("Energy Spent with an AI: {:.0f}".format(env.total_energy_ai))
print("Energy Spent with no AI: {:.0f}".format(env.total_energy_noai))

energy_saved = env.total_energy_noai - env.total_energy_ai
print("ENERGY SAVED: {:.0f} %".format(energy_saved / env.total_energy_noai * 100))
