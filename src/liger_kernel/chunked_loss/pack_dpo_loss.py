import torch
import torch.nn.functional as F
from functools import partial

from liger_kernel.chunked_loss.fused_linear_preference import LigerFusedLinearPreferenceBase


def unpack_sequence(packed: torch.Tensor,
                    num_tokens,
                    dim=1):
    if isinstance(num_tokens, torch.Tensor):
        num_tokens = num_tokens.tolist()
    sequences = torch.split(packed, num_tokens, dim=dim)
    return sequences


def chunk_nested_list(nested_list, chunk_count):
    chunk_size = len(nested_list) // chunk_count
    return [nested_list[i:i + chunk_size] for i in range(0, len(nested_list), chunk_size)]


def chunk_to_tensor(chosen_nested_list, rejected_nested_list):
    # chosen 拼接在前面，rejected 拼接在后面
    out_tensor = [torch.cat(chosen_nested_list, dim=1), torch.cat(rejected_nested_list, dim=1)]
    len_chosen = out_tensor[0].shape[1]
    return torch.cat(out_tensor, dim=1), len_chosen


class LigerFusedLinearPackDPOFunction(LigerFusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(
            chosen_logps,
            rejected_logps,
            chosen_bs,
            ref_chosen_logps=None,
            ref_rejected_logps=None,
            beta=0.1,
            loss_type='sigmoid',
            loss_weight=1.0,
            runnings=None,
    ):
        """
        Paper: https://arxiv.org/pdf/2305.18290

        Formula:
        L_DPO = -E[ log_sigmoid( β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))) ) ]

        Where:
        - π(y|x): Policy (model) probability
        - π_ref(y|x): Reference model probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - β: Weight for the direct preference loss
        - E: Expected value over the dataset

        Args:
            chosen_logps: Log probabilities of chosen tokens (batch_size,)
            rejected_logps: Log probabilities of rejected tokens (batch_size,)
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the direct preference loss
        """
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        chosen_rewards = beta * (
                chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (
                rejected_logps - ref_rejected_logps)

        if loss_type == 'sigmoid':
            logits_diff = beta * logits
            loss = -F.logsigmoid(logits_diff).sum() / chosen_bs

        elif loss_type == 'bco_pair':
            delta = runnings.mean
            loss = -F.logsigmoid(chosen_rewards - delta) - F.logsigmoid(-(rejected_rewards - delta))
            loss = loss.sum() / chosen_bs
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss, chosen_rewards, rejected_rewards

    @staticmethod
    def forward(
            ctx,
            _input,
            weight,
            target,
            bias=None,
            ref_input=None,
            ref_weight=None,
            ref_bias=None,
            ignore_index=-100,
            beta=0.1,
            rpo_alpha=1.0,
            loss_types=('sigmoid',),
            loss_weights=(1.0,),
            runnings=None,
            compute_nll_loss=False,
            compiled=True,
            use_ref_model=True,
            num_tokens=None,
            global_chosen_label_sum=None,
            **loss_kwargs,
    ):

        CHUNK_SIZE = 1  # TODO 暂时只支持 1

        # Gradients to be accumulated
        grad_weight = torch.zeros_like(weight)
        grad_chosen_inputs = []
        grad_rejected_inputs = []
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        # Loss to be accumulated
        loss_acc = torch.zeros((), device=_input.device)

        # Metrics to be recorded
        policy_chosen_logits_mean = []
        policy_rejected_logits_mean = []
        bco_pair_rewards_batch = []
        policy_nll_loss = torch.zeros((), device=_input.device)

        # TODO: 为了防止 sp 情况下额外 pad 导致的错误，需要先移除 pad 部分
        # TODO 计算 global 值,在 sp 情况下可能有 pad，先移除
        target_ = unpack_sequence(target, num_tokens)
        if global_chosen_label_sum is None:
            global_chosen_label_sum = sum([(t != ignore_index).sum() for t in target_[0::2]])

        compute_loss = partial(
            LigerFusedLinearPackDPOFunction._compute_loss,
            preference_loss_fn=LigerFusedLinearPackDPOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            alpha=rpo_alpha,
            beta=beta,
            loss_types=loss_types,
            loss_weights=loss_weights,
            runnings=runnings,
            compute_nll_loss=compute_nll_loss,
            num_tokens=num_tokens,
            global_chosen_label_sum=global_chosen_label_sum,
            use_ref_model=use_ref_model,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            **loss_kwargs
        )

        def fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk):
            """
            Fused forward and backward pass for a chunk of input and target.
            """
            if bias is not None:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1, 3), has_aux=True)(
                    input_chunk,
                    weight,
                    target_chunk,
                    bias,
                    ref_input_chunk=ref_input_chunk,
                    len_chosen_chunk=len_chosen_chunk
                )
            else:
                return torch.func.grad_and_value(compute_loss, argnums=(0, 1), has_aux=True)(
                    input_chunk, weight, target_chunk, ref_input_chunk=ref_input_chunk,
                    len_chosen_chunk=len_chosen_chunk
                )

        def accumulate_chunk(input_chunk, target_chunk, ref_input_chunk=None, len_chosen_chunk=None):
            if bias is not None:
                (
                    (chunk_grad_input, chunk_grad_weight, chunk_grad_bias),
                    (
                        chunk_loss, (
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            bco_pair_rewards,
                            chunk_nll_loss)
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk)
                grad_bias.add_(chunk_grad_bias)  # accumulate bias gradient
            else:
                (
                    (chunk_grad_input, chunk_grad_weight),
                    (
                        chunk_loss, (
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            bco_pair_rewards,
                            chunk_nll_loss)
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk)

            # Accumulate gradients
            # print(chunk_grad_weight.dtype,'xxxx')
            grad_weight.add_(chunk_grad_weight)
            # TODO: 暂时只支持 chunk_size=1，否则这里是错误的
            grad_chosen_inputs.append(chunk_grad_input[:, :len_chosen_chunk])
            grad_rejected_inputs.append(chunk_grad_input[:, len_chosen_chunk:])

            # Accumulate loss
            loss_acc.add_(chunk_loss)

            # Accumulate metrics
            policy_chosen_logits_mean.append(chunk_chosen_logits_mean)
            policy_rejected_logits_mean.append(chunk_rejected_logits_mean)
            policy_nll_loss.add_(chunk_nll_loss)

            if bco_pair_rewards.sum() != 0:
                bco_pair_rewards_batch.append(bco_pair_rewards)

        if compiled:
            fused_fwd_bwd = torch.compile(fused_fwd_bwd)

        _input = unpack_sequence(_input, num_tokens)
        ref_input = unpack_sequence(ref_input, num_tokens)

        chunks = max(1, len(_input) // (2 * CHUNK_SIZE))
        _chosen_input_chunks = chunk_nested_list(_input[0::2], chunks)
        _chosen_target_chunks = chunk_nested_list(target_[0::2], chunks)
        _rejected_input_chunks = chunk_nested_list(_input[1::2], chunks)
        _rejected_target_chunks = chunk_nested_list(target_[1::2], chunks)

        if use_ref_model:
            _ref_chosen_input_chunks = chunk_nested_list(ref_input[0::2], chunks)
            _ref_rejected_input_chunks = chunk_nested_list(ref_input[1::2], chunks)

        for (
                chosen_input_chunk,
                rejected_input_chunk,
                chosen_target_chunk,
                rejected_target_chunk,
                ref_chosen_input_chunk,
                ref_rejected_input_chunk,
        ) in zip(
            _chosen_input_chunks,
            _rejected_input_chunks,
            _chosen_target_chunks,
            _rejected_target_chunks,
            (_ref_chosen_input_chunks if use_ref_model else [None] * len(_chosen_input_chunks)),
            (_ref_rejected_input_chunks if use_ref_model else [None] * len(_rejected_input_chunks)),
            strict=False,
        ):
            input_chunk, len_chosen_chunk = chunk_to_tensor(chosen_input_chunk, rejected_input_chunk)
            ref_input_chunk, _ = chunk_to_tensor(ref_chosen_input_chunk,
                                                 ref_rejected_input_chunk) if use_ref_model else None
            target_chunk, _ = chunk_to_tensor(chosen_target_chunk, rejected_target_chunk)

            # mark input_chunk, target_chunk, and target dimension 1 as dynamic to prevent torch.compile recompilation
            torch._dynamo.mark_dynamic(input_chunk, 1)
            torch._dynamo.mark_dynamic(target_chunk, 1)
            # torch._dynamo.mark_dynamic(target, 1)
            torch._dynamo.mark_dynamic(ref_input_chunk, 1) if use_ref_model else None

            # accumulate loss, gradients, and metrics
            accumulate_chunk(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk)

        # 更新 bco 的 running
        if len(bco_pair_rewards_batch) > 0:
            bco_pair_rewards_batch = torch.cat(bco_pair_rewards_batch).mean().detach()
            runnings.update(bco_pair_rewards_batch)

        grad_inputs = [torch.cat([grad_chosen_input, grad_rejected_input], dim=1) for
                       grad_chosen_input, grad_rejected_input in zip(grad_chosen_inputs, grad_rejected_inputs)]
        grad_inputs = torch.cat(grad_inputs, dim=1)

        policy_chosen_logits_mean = torch.stack(policy_chosen_logits_mean, dim=0)
        policy_rejected_logits_mean = torch.stack(policy_rejected_logits_mean, dim=0)

        ctx.save_for_backward(
            grad_inputs,
            grad_weight,
            grad_bias,
        )

        reward_margin = (policy_chosen_logits_mean - policy_rejected_logits_mean).mean()
        reward_acc = (policy_chosen_logits_mean > policy_rejected_logits_mean).float().mean()

        return_vars = (
            policy_chosen_logits_mean.mean(),
            policy_rejected_logits_mean.mean(),
            reward_margin,
            reward_acc,
            policy_nll_loss,
        )
        return loss_acc, *return_vars

    @staticmethod
    def _compute_loss(
            input_chunk,
            weight,
            target_chunk,
            bias=None,
            preference_loss_fn=None,
            ignore_index=-100,
            alpha=1.0,
            beta=0.1,
            loss_types=('sigmoid',),
            loss_weights=(1.0,),
            runnings=None,
            compute_nll_loss=True,
            use_ref_model=False,
            ref_input_chunk=None,
            ref_weight=None,
            ref_bias=None,
            len_chosen_chunk=None,
            num_tokens=None,
            global_chosen_label_sum=None,
            **loss_kwargs,
    ):
        """
        Compute the total loss for a chunk of input and target, while using an alignment/preference loss function.
        Args:
            preference_loss_fn (callable): Loss function to compute the loss on a chunk of input/target.
            input_chunk (torch.Tensor): Chunk of input tensor. Shape: (2 * chunk_size, sequence_length, hidden_size).
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size).
            target_chunk (torch.Tensor): Chunk of target tensor. Shape: (2 * chunk_size, sequence_length).
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,).
            full_target (torch.Tensor): Full target tensor. Shape: (batch_size, sequence_length).
            ignore_index (int): Index to ignore for loss computation.
            alpha (float): Weight for the NLL loss.
            beta (float): Weight for the preference loss.
            compute_nll_loss (bool): Whether to compute NLL loss.
            use_ref_model (bool): Whether to use a reference model for the alignment loss.
            ref_weight (torch.Tensor): Reference weight tensor. Shape: (vocab_size, hidden_size).
            ref_bias (torch.Tensor, optional): Reference bias tensor. Shape: (vocab_size,).
            loss_kwargs (dict): Additional arguments for the loss function.
        """
        (
            chosen_logps,
            rejected_logps,
            # chosen_logits,
            # rejected_logits,
            chosen_nll_loss,
        ) = LigerFusedLinearPackDPOFunction.chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            bias=bias,
            ignore_index=ignore_index,
            compute_nll_loss=compute_nll_loss,
            len_chosen_chunk=len_chosen_chunk,
        )
        chosen_bs = len(num_tokens) // 2
        # 是否要矫正，需要商榷,如果不全局矫正，则需要在外面把 all-reduce 关掉
        chosen_nll_loss = chosen_nll_loss / global_chosen_label_sum

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    # ref_chosen_logits,
                    # ref_rejected_logits,
                    ref_chosen_nll_loss,
                ) = LigerFusedLinearPackDPOFunction.chunk_forward(
                    ref_input_chunk,
                    ref_weight,
                    target_chunk,
                    ref_bias,
                    ignore_index=ignore_index,
                    len_chosen_chunk=len_chosen_chunk,
                    compute_nll_loss=False,  # We don't need NLL loss for the reference model
                )
            loss_kwargs["ref_chosen_logps"] = ref_chosen_logps
            loss_kwargs["ref_rejected_logps"] = ref_rejected_logps

        bco_pair_rewards = torch.tensor(0)
        dpo_losses, chosen_rewards, rejected_rewards = 0, 0, 0
        for curr_type, curr_weight in zip(loss_types, loss_weights):
            curr_losses, curr_chosen_rewards, curr_rejected_rewards = preference_loss_fn(
                chosen_logps, rejected_logps, chosen_bs, beta=beta, loss_type=curr_type,
                loss_weight=curr_weight, runnings=runnings, **loss_kwargs
            )
            if curr_type == 'bco_pair':
                # 必须要是 batch 级别更新，否则不准确，因此需要特别处理
                bco_pair_rewards = torch.cat((curr_chosen_rewards, curr_rejected_rewards), 0).detach()

            dpo_losses = dpo_losses + curr_losses * curr_weight
            chosen_rewards = chosen_rewards + curr_chosen_rewards * curr_weight
            rejected_rewards = rejected_rewards + curr_rejected_rewards * curr_weight

        loss = alpha * chosen_nll_loss + dpo_losses
        return_vars = (
            chosen_rewards.mean(),
            rejected_rewards.mean(),
            bco_pair_rewards,
            alpha * chosen_nll_loss,
        )
        return loss, return_vars

    @staticmethod
    def chunk_forward(
            input_chunk,
            weight,
            target_chunk,
            bias=None,
            ignore_index=-100,
            compute_nll_loss=True,
            len_chosen_chunk=None
    ):
        logits_chunk = input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias
        log_probs_chunk = F.log_softmax(logits_chunk.float(), dim=-1)

        chosen_nll_loss = 0.0
        if compute_nll_loss:
            chosen_nll_loss = F.nll_loss(
                log_probs_chunk[:, :len_chosen_chunk].view(-1, log_probs_chunk.shape[-1]),
                target_chunk[:, :len_chosen_chunk].view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        loss_mask = target_chunk != ignore_index
        label_chunk = torch.where(loss_mask, target_chunk, 0)

        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)

        style = 2
        if style == 1:
            # old 写法
            # TODO 这一步应该不对，dpo 并没有  / loss_mask.sum(-1)
            sum_logps = per_token_logps * loss_mask
            chosen_logps = sum_logps[:, :len_chosen_chunk].sum(-1) / loss_mask[:, :len_chosen_chunk].sum(-1)
            rejected_logps = sum_logps[:, len_chosen_chunk:].sum(-1) / loss_mask[:, len_chosen_chunk:].sum(-1)
        else:
            sum_logps = per_token_logps * loss_mask
            chosen_logps = sum_logps[:, :len_chosen_chunk].sum(-1)
            rejected_logps = sum_logps[:, len_chosen_chunk:].sum(-1)

        return (
            chosen_logps,
            rejected_logps,
            # chosen_logits,
            # rejected_logits,
            chosen_nll_loss,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        # print(grads[1].dtype,'xyyyyyyyxx')
        return *grads, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class LigerFusedLinearPackDPOLoss(torch.nn.Module):
    """
    Fused linear layer with DPO loss.
    """

    def __init__(
            self,
            ignore_index: int = -100,
            beta: float = 0.1,
            rpo_alpha: float = 1.0,
            loss_types: tuple = ('sigmoid',),
            loss_weights: tuple = (1.0,),
            runnings=None,
            compiled: bool = True,
            use_ref_model: bool = True,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.rpo_alpha = rpo_alpha
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.runnings = runnings
        self.compute_nll_loss = rpo_alpha > 0
        self.compiled = compiled
        self.use_ref_model = use_ref_model

    def forward(
            self,
            lin_weight,
            _input,
            target,
            bias=None,
            ref_input=None,
            ref_weight=None,
            ref_bias=None,
            num_tokens=None,
            global_chosen_label_sum=None,
    ):
        return LigerFusedLinearPackDPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.beta,
            self.rpo_alpha,
            self.loss_types,
            self.loss_weights,
            self.runnings,
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
            num_tokens,
            global_chosen_label_sum
        )
