import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import distrax

class CausalRLAgent(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_size)(x)

def select_action(params, state, agent, key):
    logits = agent.apply(params, state)
    dist = distrax.Categorical(logits=logits)
    action = dist.sample(seed=key)
    log_prob = dist.log_prob(action)
    return int(action), log_prob

def train_causal_rl(agent, params, environment, episodes, gamma=0.99):
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(0)

    for episode in range(episodes):
        state = environment.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            key, subkey = jax.random.split(key)
            state_jnp = jnp.array(state)
            action, log_prob = select_action(params, state_jnp, agent, subkey)
            next_state, reward, done, _ = environment.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = jnp.array(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - jnp.mean(discounted_rewards)) / (jnp.std(discounted_rewards) + 1e-9)

        # Calculate loss
        def loss_fn(params):
            logits = [agent.apply(params, jnp.array(s)) for s in states]
            log_probs = [distrax.Categorical(logits=logit).log_prob(a) for logit, a in zip(logits, actions)]
            return -jnp.sum(jnp.array(log_probs) * discounted_rewards)

        states = [jnp.array(s) for s in rewards[:-1]]  # Assuming states collected during rollout
        actions = [a for a in rewards[:-1]]            # Actions collected during rollout

        # Backpropagate and optimize
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    return params