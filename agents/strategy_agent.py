from dataclasses import dataclass
from typing import List, Union
import gc
import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer
from .utils import find_executable_batch_size, mellowmax


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        print(message)


@dataclass
class StrategyConfig:
    num_steps: int = 500
    optim_str_init: Union[str, List[str]] = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    search_width: int = 64
    batch_size: int = None
    topk: int = 2048
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = False
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    mu: float = 0.4


@dataclass
class StrategyResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


class StrategyAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: StrategyConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else self._get_nonascii_toks(tokenizer, device=model.device)
        )
        self.prefix_cache = None
        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            print(
                f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if model.device == torch.device("cpu"):
            print(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not tokenizer.chat_template:
            print(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )

    def _get_nonascii_toks(self, tokenizer, device="cpu"):
        def is_ascii(s):
            return s.isascii() and s.isprintable()

        nonascii_toks = []
        for i in range(tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                nonascii_toks.append(i)
        
        if tokenizer.bos_token_id is not None:
            nonascii_toks.append(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            nonascii_toks.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            nonascii_toks.append(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            nonascii_toks.append(tokenizer.unk_token_id)
        
        return torch.tensor(nonascii_toks, device=device)

    def execute(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ) -> StrategyResult:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = messages.copy()

        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        self.before_str = before_str

        target = " " + target if self.config.add_space_before_target else target

        before_ids = self.tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device, torch.int64)
        after_ids = self.tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)
        target_ids = self.tokenizer([target], add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device, torch.int64)

        embedding_layer = self.embedding_layer
        before_embeds, after_embeds, target_embeds = [
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        ]

        if self.config.use_prefix_cache:
            with torch.no_grad():
                output = self.model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        return self._run_attack_loop()

    def _run_attack_loop(self) -> StrategyResult:
        buffer = AttackBuffer(self.config.buffer_size)

        if isinstance(self.config.optim_str_init, str):
            init_optim_ids = self.tokenizer(
                self.config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.model.device)
            if self.config.buffer_size > 1:
                init_buffer_ids = (
                    self.tokenizer(
                        [".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", 
                         "@", "#", "$", "%", "&", "*", "w", "x", "y", "z"],
                        add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(self.model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (self.config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:
            if len(self.config.optim_str_init) != self.config.buffer_size:
                print(
                    f"Using {len(self.config.optim_str_init)} initializations but buffer size is set to {self.config.buffer_size}"
                )
            try:
                init_buffer_ids = self.tokenizer(
                    self.config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.model.device)
            except ValueError:
                print(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, self.config.buffer_size)

        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.before_embeds.repeat(true_buffer_size, 1, 1),
                    self.embedding_layer(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        init_buffer_losses = find_executable_batch_size(
            lambda bs, embeds: self._compute_loss(
                bs, embeds, self.target_ids, self.prefix_cache, self.config.use_mellowmax, self.config.mellowmax_alpha
            ),
            true_buffer_size
        )(init_buffer_embeds)

        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(self.tokenizer)
        print("Initialized attack buffer.")

        optim_ids = buffer.get_best_ids()
        losses = []
        optim_strings = []
        momentum_grad = None

        for _ in range(self.config.num_steps):
            optim_ids_onehot_grad = self._compute_token_gradient(
                optim_ids,
                self.before_embeds,
                self.after_embeds,
                self.target_embeds,
                self.target_ids,
                self.prefix_cache,
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

                sampled_ids = self._sample_candidates(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    self.config.search_width,
                    self.before_str,
                    self.config.topk,
                    self.config.n_replace,
                )

                if self.config.filter_ids:
                    sampled_ids = self._filter_candidates(sampled_ids)

                new_search_width = sampled_ids.shape[0]

                batch_size = (
                    new_search_width if self.config.batch_size is None else self.config.batch_size
                )

                if self.prefix_cache:
                    input_embeds = torch.cat(
                        [
                            self.embedding_layer(sampled_ids),
                            self.after_embeds.repeat(new_search_width, 1, 1),
                            self.target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:
                    input_embeds = torch.cat(
                        [
                            self.before_embeds.repeat(new_search_width, 1, 1),
                            self.embedding_layer(sampled_ids),
                            self.after_embeds.repeat(new_search_width, 1, 1),
                            self.target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

                loss = self._compute_loss(
                    batch_size,
                    input_embeds,
                    self.target_ids,
                    self.prefix_cache,
                    self.config.use_mellowmax,
                    self.config.mellowmax_alpha,
                )

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(self.tokenizer)

            if self.stop_flag:
                print("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        return StrategyResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

    def _compute_loss(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        target_ids: Tensor,
        prefix_cache=None,
        use_mellowmax: bool = False,
        mellowmax_alpha: float = 1.0,
    ) -> Tensor:
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if prefix_cache:
                    if (
                        not prefix_cache_batch
                        or current_batch_size != search_batch_size
                    ):
                        prefix_cache_batch = [
                            [
                                x.expand(current_batch_size, -1, -1, -1)
                                for x in prefix_cache[i]
                            ]
                            for i in range(len(prefix_cache))
                        ]

                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                        use_cache=True,
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)

                if use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    loss = mellowmax(
                        -label_logits, alpha=mellowmax_alpha, dim=-1
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _compute_token_gradient(
        self,
        optim_ids: Tensor,
        before_embeds: Tensor,
        after_embeds: Tensor,
        target_embeds: Tensor,
        target_ids: Tensor,
        prefix_cache=None,
        use_mellowmax: bool = False,
        mellowmax_alpha: float = 1.0,
    ) -> Tensor:
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=self.embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(self.model.device, self.model.dtype)
        optim_ids_onehot.requires_grad_()

        optim_embeds = optim_ids_onehot @ self.embedding_layer.weight

        if prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, after_embeds, target_embeds], dim=1
            )
            output = self.model(
                inputs_embeds=input_embeds,
                past_key_values=prefix_cache,
                use_cache=True,
            )
        else:
            input_embeds = torch.cat(
                [
                    before_embeds,
                    optim_embeds,
                    after_embeds,
                    target_embeds,
                ],
                dim=1,
            )
            output = self.model(inputs_embeds=input_embeds)

        logits = output.logits

        shift = input_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[
            ..., shift - 1 : -1, :
        ].contiguous()
        shift_labels = target_ids

        if use_mellowmax:
            label_logits = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def _sample_candidates(
        self,
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        before_str: str,
        topk: int = 256,
        n_replace: int = 1,
    ) -> Tensor:
        if self.not_allowed_ids is not None:
            grad[:, self.not_allowed_ids.to(grad.device)] = float("inf")

        topk_ids = (-grad).topk(topk, dim=1).indices

        new_ids = torch.tensor([]).to(torch.int64).to(self.model.device)

        selected_indices = torch.randperm(topk_ids.size(1))[:search_width]
        selected_tokens = topk_ids[0, selected_indices].unsqueeze(1)
        before_str = self.tokenizer.batch_decode(selected_tokens)

        for i in range(topk_ids.shape[0]):
            next_token_ids = self._predict_next_tokens(before_str, top_k=20)
            
            topk_ids_expanded = topk_ids[i].view(1, -1)
            matches = (next_token_ids.unsqueeze(2) == topk_ids_expanded).any(dim=2)

            matches_numeric = matches.to(torch.float)
            first_match_idx = matches_numeric.argmax(dim=1)
            has_match = matches.any(dim=1)

            next_tokens_id = torch.where(
                has_match,
                next_token_ids[
                    torch.arange(search_width), first_match_idx
                ],
                ids[i],
            )

            before_str = [
                a + b for a, b in zip(before_str, self.tokenizer.batch_decode(next_tokens_id))
            ]
            new_ids = torch.cat((new_ids, next_tokens_id.unsqueeze(1)), dim=1)

        return new_ids

    def _predict_next_tokens(self, input_texts: List[str], top_k: int = 10) -> Tensor:
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
            logits = outputs.logits

        last_token_logits = logits[:, -1, :]
        top_k_logits, top_k_indices = torch.topk(
            last_token_logits, k=top_k, dim=-1
        )
        return top_k_indices

    def _filter_candidates(self, ids: Tensor) -> Tensor:
        ids_decoded = self.tokenizer.batch_decode(ids)
        filtered_ids = []

        for i in range(len(ids_decoded)):
            ids_encoded = self.tokenizer(
                ids_decoded[i], return_tensors="pt", add_special_tokens=False
            ).to(ids.device)["input_ids"][0]
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

        if not filtered_ids:
            raise RuntimeError(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different initialization"
            )

        return torch.stack(filtered_ids)
