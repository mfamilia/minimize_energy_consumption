import numpy as np

class DQN(object):
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])

        if (len(self.memory) > self.max_memory):
            self.memory.pop(0)

    def get_batch(self, model, batch_size = 10):
        length_memory = len(self.memory)
        number_inputs = self.memory[0][0][0].shape[1]
        number_outputs = model.output_shape[-1] 
        current_batch_size = min(length_memory, batch_size)
        inputs = np.zeros((current_batch_size, number_inputs))
        targets = np.zeros((current_batch_size, number_outputs))

        for i, idx in enumerate(np.random.randint(0, length_memory, size = current_batch_size)):
            selected_memory = self.memory[idx]
            current_state, action, reward, next_state = selected_memory[0]
            game_over = selected_memory[1] 
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            q_state_action = np.max(model.predict(next_state)[0])

            if (game_over):
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * q_state_action

        return inputs, targets