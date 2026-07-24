from adapters import ENV_REGISTRY
from adapters.registry import get_env_class
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from eval.runner import run_evaluation


P0_ENVS = ("connect_four", "goofspiel", "leduc_poker")


def test_registry_contains_scaffold_environments():
    for env_id in P0_ENVS:
        assert env_id in ENV_REGISTRY


def test_random_agents_complete_scaffold_games():
    for env_id in P0_ENVS:
        env = get_env_class(env_id)(seed=7)
        agent = RandomAgent(seed=11)
        opponent = RandomAgent(seed=13)
        obs = env.reset()

        for _ in range(200):
            if obs.done:
                break
            current = agent if obs.current_player == 0 else opponent
            action = current.act(obs)
            assert action in obs.legal_actions
            obs = env.step(action)

        assert obs.done
        assert obs.outcome is not None


def test_heuristic_agent_handles_all_action_vocabularies():
    for env_id in P0_ENVS:
        env = get_env_class(env_id)(seed=17)
        agent = HeuristicAgent(seed=19)
        obs = env.reset()

        for _ in range(200):
            if obs.done:
                break
            action = agent.act(obs)
            assert action in obs.legal_actions
            obs = env.step(action)

        assert obs.done


def test_runner_evaluates_each_scaffold_environment():
    for env_id in P0_ENVS:
        result = run_evaluation(
            environment=env_id,
            agent_name="heuristic",
            opponents=["random"],
            seed=23,
            episodes=2,
            max_steps=200,
        )

        assert result["summary"]["environment_steps"] > 0
        assert result["per_opponent"]["random"]["num_episodes"] == 2
