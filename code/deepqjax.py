import os
import time
import jax
import jax.numpy as jnp

import chex
import flax
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import wandb
import numpy as np
import matplotlib.pyplot as plt


import gymnax
import flashbax as fbx
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

def make_train(config):
    config['num_updates'] = config['total_timesteps'] // config['num_envs']

    basic_env, env_params = gymnax.make(config['env_name'])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config['num_envs'])(_rng)

        buffer = fbx.make_flat_buffer(
            max_length=config['buffer_size'],
            min_length=config['buffer_batch_size'],
            sample_batch_size=config['buffer_batch_size'],
            add_sequences=False,
            add_batch_size=config['num_envs']
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample)
        )

        rng = jax.random.PRNGKey(0)
        _action = basic_env.action_space().sample(rng)
        _, _env_state = env.reset(rng, env_params)
        _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        network = QNetwork(action_dim=env.action_space(env_params).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = 1.0 - (count / config['num_updates'])
            return config['learning_rate'] * frac

        lr = linear_schedule if config.get("lr_linear_decay", False) else config['learning_rate']
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree_map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0
        )

        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(rng, 2)
            eps = jnp.clip(
                ((config["eps_finish"] - config["eps_start"]) / config["eps_ann_time"]) * t
                + config["eps_start"],
                config["eps_finish"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)
            choosed_actions = jnp.where(
                jax.random.uniform(rng_a, shape=greedy_actions.shape) < eps,
                jax.random.randint(rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]),
                greedy_actions,
            )
            return choosed_actions

        def _update_step(runner_state, _):
            train_state, buffer_state, env_state, last_obs, rng = runner_state

            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(rng_a, q_vals, train_state.timesteps)

            obs, env_state, reward, done, info = vmap_step(config["num_envs"])(rng_s, env_state, action)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config['num_envs']
            )

            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            def _learn_phase(train_state, rng):
                learn_batch = buffer.sample(buffer_state, rng).experience
                q_next_target = network.apply(train_state.target_network_params, learn_batch.second.obs)
                q_next_target = jnp.max(q_next_target, axis=-1)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config['gamma'] * q_next_target
                )

                def _loss_fn(params):
                    q_vals = network.apply(params, learn_batch.first.obs)
                    chosen_qvals = jnp.take_along_axis(
                        q_vals, jnp.expand_dims(learn_batch.first.action, axis=-1), axis=-1
                    ).squeeze(axis=-1)
                    return jnp.mean(jnp.square(target - chosen_qvals))

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (train_state.timesteps > config["learning_starts"])
                & (train_state.timesteps % config["training_interval"] == 0)
            )

            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda ts, rng: _learn_phase(ts, rng),
                lambda ts, rng: (ts, jnp.array(0.0)),
                train_state,
                rng,
            )

            train_state = jax.lax.cond(
                train_state.timesteps % config["target_update_interval"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["tau"]
                    )
                ),
                lambda ts: ts,
                train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                "returns": info["returned_episode_returns"].mean(),
            }

            if config.get("WANDB_MODE", "disabled") == "online":
                def callback(metrics):
                    if metrics["timesteps"] % 100 == 0:
                        wandb.log(metrics)
                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, env_state, obs, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)


        start_time = time.time()
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=config["num_updates"]
        )
        total_time = time.time() - start_time

        sps = config['total_timesteps'] / total_time

        def print_final_results(metrics_dict, total_time, sps):
            timesteps = np.array(metrics_dict['timesteps'])
            returns = np.array(metrics_dict['returns'])
            updates = np.array(metrics_dict['updates'])
            loss = np.array(metrics_dict['loss'])

            final_timesteps = timesteps[-1]
            final_updates = updates[-1]
            final_loss = loss[-1]
            final_returns = returns[-1]

            print(f"\n=== Training Complete ===")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Steps/Second: {sps:.2f}")
            print(f"Final Timesteps: {final_timesteps}")
            print(f"Final Updates: {final_updates}")
            print(f"Final Loss: {final_loss:.4f}")
            print(f"Final Returns: {final_returns:.2f}")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            ax1.plot(timesteps, returns)
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Average Returns')
            ax1.set_title('Learning Progress')
            ax1.grid(True)

            valid_loss_idx = loss > 0
            if np.any(valid_loss_idx):
                ax2.plot(updates[valid_loss_idx], loss[valid_loss_idx])
                ax2.set_xlabel('Updates')
                ax2.set_ylabel('Loss')
                ax2.set_title('Training Loss')
                ax2.grid(True)

            ax3.plot(timesteps, returns)
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Returns')
            ax3.set_title('Returns Over Time')
            ax3.grid(True)

            if len(returns) > 100:
                returns_ma = np.convolve(returns, np.ones(100)/100, mode='valid')
                ax4.plot(timesteps[:len(returns_ma)], returns_ma)
                ax4.set_xlabel('Timesteps')
                ax4.set_ylabel('Moving Average (100)')
                ax4.set_title('Smoothed Returns')
                ax4.grid(True)

            plt.tight_layout()
            plt.show()

        jax.debug.callback(print_final_results, metrics, total_time, sps)

        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "final_metrics": {
                'total_time': total_time,
                'steps_per_second': sps,
                'final_timesteps': metrics['timesteps'][-1],
                'final_updates': metrics['updates'][-1],
                'final_loss': metrics['loss'][-1],
                'final_returns': metrics['returns'][-1],
            }
        }




    return train

def main():

    config = {
        "num_envs": 4,
        "buffer_size": 10000,
        "buffer_batch_size": 128,
        "total_timesteps": int(4e5),
        "eps_start": 1.0,
        "eps_finish": 0.05,
        "eps_ann_time": int(1.5e5),
        "target_update_interval": 500,
        "learning_rate": 4.561407e-4,
        "learning_starts": 1000,
        "training_interval": 2,
        "lr_linear_decay": False,
        "gamma": 0.99,
        "tau": 1,
        "env_name": "MountainCar-v0",
        "seed": 0,
        "num_seeds": 1,
        "WANDB_MODE": "disabled",
        "ENTITY": "",
        "PROJECT": "",
    }
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", config["env_name"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_dqn_{config["env_name"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["seed"])
    rngs = jax.random.split(rng, config["num_seeds"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

if __name__ == "__main__":
    main()

