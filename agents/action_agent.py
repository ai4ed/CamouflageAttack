from typing import List, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from .strategy_agent import StrategyAgent, StrategyConfig, StrategyResult
from .camouflage_agent import CamouflageAgent


class ActionAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: StrategyConfig = None,
    ):
        if config is None:
            config = StrategyConfig()

        self.strategy_agent = StrategyAgent(model, tokenizer, config)
        self.camouflage_agent = CamouflageAgent(
            model,
            tokenizer,
            model.get_input_embeddings(),
            self.strategy_agent.not_allowed_ids
        )
        self.config = config

    def execute(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> StrategyResult:
        self.strategy_agent.execute(messages, target)
        
        buffer = self.camouflage_agent.init_attack_buffer(
            self.config,
            self.strategy_agent.before_embeds,
            self.strategy_agent.after_embeds,
            self.strategy_agent.target_embeds,
            self.strategy_agent.target_ids,
            self.strategy_agent.prefix_cache,
        )
        
        optim_ids = buffer.get_best_ids()
        losses = []
        optim_strings = []
        momentum_grad = None

        for _ in range(self.config.num_steps):
            optim_ids_onehot_grad = self.camouflage_agent.compute_token_gradient(
                optim_ids,
                self.strategy_agent.before_embeds,
                self.strategy_agent.after_embeds,
                self.strategy_agent.target_embeds,
                self.strategy_agent.target_ids,
                self.strategy_agent.prefix_cache,
                self.config.use_mellowmax,
                self.config.mellowmax_alpha,
            )

            mu = self.config.mu
            with torch.no_grad():
                if momentum_grad is None:
                    momentum_grad = optim_ids_onehot_grad
                else:
                    momentum_grad = momentum_grad * mu + optim_ids_onehot_grad * (1 - mu)
                    optim_ids_onehot_grad = momentum_grad.clone()

                sampled_ids = self.camouflage_agent.sample_candidates(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    self.config.search_width,
                    self.strategy_agent.before_str,
                    self.config.topk,
                    self.config.n_replace,
                )

                if self.config.filter_ids:
                    sampled_ids = self.camouflage_agent.filter_candidates(sampled_ids)

                new_search_width = sampled_ids.shape[0]

                batch_size = (
                    new_search_width if self.config.batch_size is None else self.config.batch_size
                )

                if self.strategy_agent.prefix_cache:
                    input_embeds = torch.cat(
                        [
                            self.camouflage_agent.embedding_layer(sampled_ids),
                            self.strategy_agent.after_embeds.repeat(new_search_width, 1, 1),
                            self.strategy_agent.target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            self.strategy_agent.before_embeds.repeat(new_search_width, 1, 1),
                            self.camouflage_agent.embedding_layer(sampled_ids),
                            self.strategy_agent.after_embeds.repeat(new_search_width, 1, 1),
                            self.strategy_agent.target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                loss = self.camouflage_agent.compute_loss(
                    batch_size,
                    input_embeds,
                    self.strategy_agent.target_ids,
                    self.strategy_agent.prefix_cache,
                    self.config.use_mellowmax,
                    self.config.mellowmax_alpha,
                )

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.camouflage_agent.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(self.camouflage_agent.tokenizer)

            if self.strategy_agent.stop_flag:
                print("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        return StrategyResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )
