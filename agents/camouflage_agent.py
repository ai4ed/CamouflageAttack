import gc
import torch
from torch import Tensor
from typing import List, Tuple
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


class CamouflageAgent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        embedding_layer: torch.nn.Embedding,
        not_allowed_ids: Tensor = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.not_allowed_ids = not_allowed_ids

    def init_attack_buffer(
        self,
        config,
        before_embeds: Tensor,
        after_embeds: Tensor,
        target_embeds: Tensor,
        target_ids: Tensor,
        prefix_cache=None,
    ) -> AttackBuffer:
        print(f"Initializing attack buffer of size {config.buffer_size}...")

        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = self.tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(self.model.device)
            if config.buffer_size > 1:
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
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:
            if len(config.optim_str_init) != config.buffer_size:
                print(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = self.tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self.model.device)
            except ValueError:
                print(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        if prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    self.embedding_layer(init_buffer_ids),
                    after_embeds.repeat(true_buffer_size, 1, 1),
                    target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    before_embeds.repeat(true_buffer_size, 1, 1),
                    self.embedding_layer(init_buffer_ids),
                    after_embeds.repeat(true_buffer_size, 1, 1),
                    target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        init_buffer_losses = find_executable_batch_size(
            lambda bs, embeds: self.compute_loss(
                bs, embeds, target_ids, prefix_cache, config.use_mellowmax, config.mellowmax_alpha
            ),
            true_buffer_size
        )(init_buffer_embeds)

        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(self.tokenizer)
        print("Initialized attack buffer.")

        return buffer

    def compute_loss(
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

    def compute_token_gradient(
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

    def sample_candidates(
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

    def filter_candidates(self, ids: Tensor) -> Tensor:
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
