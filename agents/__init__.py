from .strategy_agent import StrategyAgent, StrategyConfig, StrategyResult
from .camouflage_agent import CamouflageAgent
from .action_agent import ActionAgent


def run(model, tokenizer, messages, target, config=None):
    agent = ActionAgent(model, tokenizer, config)
    return agent.execute(messages, target)


__all__ = [
    "StrategyAgent",
    "StrategyConfig", 
    "StrategyResult",
    "CamouflageAgent",
    "ActionAgent",
    "run"
]
