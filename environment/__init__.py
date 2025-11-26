from gymnasium.envs.registration import register

register(
    id="CityDrone-v0",
    entry_point="environment.custom_env:CityDroneEnv",
    max_episode_steps=500,
)
