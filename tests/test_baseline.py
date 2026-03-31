def test_baselineEnv():
    try:
        import gymnasium as gym
        from stable_baselines.common.vec_env import DummyVecEnv
        from stable_baselines.deepq.policies import MlpPolicy
        from stable_baselines import DQN

        env = gym.make('CartPole-v1')

        model = DQN(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=20)
        action, _ = model.predict(env.reset())
        env.step(action)
        return True
    except Exception as er:
        assert False, er


