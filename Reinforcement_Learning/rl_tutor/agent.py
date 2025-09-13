# rl_tutor/agent.py

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow_probability as tfp
import os
from . import config

class PPOAgent:
    """Proximal Policy Optimization Agent."""
    def __init__(self):
        self.state_size = config.STATE_SIZE
        self.action_size = config.ACTION_SIZE
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.memory = []

    def _build_actor(self):
        state_input = Input(shape=(self.state_size,))
        dense = Dense(64, activation='relu')(state_input)
        dense = Dense(64, activation='relu')(dense)
        action_probs = Dense(self.action_size, activation='softmax')(dense)
        
        model = Model(inputs=state_input, outputs=action_probs)
        model.compile(optimizer=Adam(learning_rate=config.ACTOR_LEARNING_RATE))
        return model

    def _build_critic(self):
        state_input = Input(shape=(self.state_size,))
        dense = Dense(64, activation='relu')(state_input)
        dense = Dense(64, activation='relu')(dense)
        state_value = Dense(1, activation='linear')(dense)

        model = Model(inputs=state_input, outputs=state_value)
        model.compile(optimizer=Adam(learning_rate=config.CRITIC_LEARNING_RATE), loss='mse')
        return model

    def remember(self, state, action, prob, reward, next_state, done):
        self.memory.append((state, action, prob, reward, next_state, done))

    def act(self, state, valid_actions, force_exploit=False):
        if not valid_actions: return None, None

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action_probs = self.actor(state)[0].numpy()
        
        masked_probs = np.zeros_like(action_probs)
        masked_probs[valid_actions] = action_probs[valid_actions]
        
        sum_masked_probs = np.sum(masked_probs)
        if sum_masked_probs > 1e-8:
            masked_probs /= sum_masked_probs
        else:
            masked_probs = np.zeros_like(action_probs)
            masked_probs[valid_actions] = 1.0 / len(valid_actions)

        if force_exploit:
            action = np.argmax(masked_probs)
            return action, 1.0

        dist = tfp.distributions.Categorical(probs=masked_probs)
        action = dist.sample().numpy()
        prob = dist.prob(action).numpy()
        
        return action, prob

    def learn(self):
        if not self.memory: return
        
        states, actions, old_probs, rewards, next_states, dones = zip(*self.memory)
        self.memory.clear()

        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(np.vstack(next_states), verbose=0).flatten()
        
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + config.GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
        
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        target_values = advantages + values
        
        for _ in range(config.EPOCHS_PER_UPDATE):
            with tf.GradientTape() as tape:
                new_probs_dist = tfp.distributions.Categorical(probs=self.actor(states))
                new_probs = new_probs_dist.prob(actions)
                ratio = new_probs / (old_probs + 1e-10)
                
                surrogate1 = ratio * advantages
                surrogate2 = tf.clip_by_value(ratio, 1.0 - config.CLIP_EPSILON, 1.0 + config.CLIP_EPSILON) * advantages
                
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                entropy_loss = -tf.reduce_mean(new_probs_dist.entropy())
                total_actor_loss = actor_loss + config.ENTROPY_BETA * entropy_loss

            actor_grads = tape.gradient(total_actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            self.critic.fit(states, target_values, epochs=1, verbose=0)
            
    def load(self):
        if os.path.exists(config.ACTOR_MODEL_PATH) and os.path.exists(config.CRITIC_MODEL_PATH):
            self.actor = load_model(config.ACTOR_MODEL_PATH)
            self.critic = load_model(config.CRITIC_MODEL_PATH)
            print("PPO models loaded successfully.")
        else:
            print("Could not find saved models, starting from scratch.")

    def save(self):
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)
        self.actor.save(config.ACTOR_MODEL_PATH)
        self.critic.save(config.CRITIC_MODEL_PATH)
        print(f"PPO models saved to {config.MODEL_DIR}/")

