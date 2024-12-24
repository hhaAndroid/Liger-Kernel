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
            full_target: Non chunked full target tensor
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the direct preference loss
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        logits_diff = beta * (chosen_logratios - rejected_logratios)
        # TODO: 分母是否正确？
        loss = -F.logsigmoid(logits_diff).sum() / chosen_bs
        return loss

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
        policy_chosen_logps = []
        policy_rejected_logps = []
        policy_chosen_logits_mean = torch.zeros((), device=_input.device)
        policy_rejected_logits_mean = torch.zeros((), device=_input.device)
        policy_nll_loss = torch.zeros((), device=_input.device)
        aggregated_aux_outputs = []  # aggregated aux outputs from all chunks

        # TODO: 为了防止 sp 情况下额外 pad 导致的错误，需要先移除 pad 部分
        # TODO 计算 global 值,在 sp 情况下可能有 pad，先移除
        target_ = unpack_sequence(target, num_tokens)
        if global_chosen_label_sum is None:
            global_chosen_label_sum = sum([(t != ignore_index).sum() for t in target_[0::2]])

        compute_loss = partial(
            LigerFusedLinearPackDPOFunction._compute_loss,
            preference_loss_fn=LigerFusedLinearPackDPOFunction.preference_loss_fn,
            ignore_index=ignore_index,
            alpha=1.0,
            beta=beta,
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
                        chunk_loss,
                        (
                            chunk_chosen_logps,
                            chunk_rejected_logps,
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            chunk_nll_loss,
                            *aux_outputs,
                        ),
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk)
                grad_bias.add_(chunk_grad_bias)  # accumulate bias gradient
            else:
                (
                    (chunk_grad_input, chunk_grad_weight),
                    (
                        chunk_loss,
                        (
                            chunk_chosen_logps,
                            chunk_rejected_logps,
                            chunk_chosen_logits_mean,
                            chunk_rejected_logits_mean,
                            chunk_nll_loss,
                            *aux_outputs,
                        ),
                    ),
                ) = fused_fwd_bwd(input_chunk, target_chunk, ref_input_chunk, len_chosen_chunk)

            # Accumulate gradients
            grad_weight.add_(chunk_grad_weight)
            # TODO: 暂时只支持 chunk_size=1，否则这里是错误的
            grad_chosen_inputs.append(chunk_grad_input[:, :len_chosen_chunk])
            grad_rejected_inputs.append(chunk_grad_input[:, len_chosen_chunk:])

            # Accumulate loss
            loss_acc.add_(chunk_loss)

            # Accumulate metrics
            policy_chosen_logps.append(chunk_chosen_logps)
            policy_rejected_logps.append(chunk_rejected_logps)
            policy_chosen_logits_mean.add_(chunk_chosen_logits_mean)
            policy_rejected_logits_mean.add_(chunk_rejected_logits_mean)
            policy_nll_loss.add_(chunk_nll_loss)

            # aux_outputs
            # Initialize storage for aux_outputs
            if len(aggregated_aux_outputs) == 0:
                for aux in aux_outputs:
                    if aux.ndim == 0:
                        aggregated_aux_outputs.append(torch.zeros((), device=aux.device))
                    else:
                        aggregated_aux_outputs.append([])

            # Process each aux_output
            for i, aux in enumerate(aux_outputs):
                if aux.ndim == 0:
                    aggregated_aux_outputs[i].add_(aux)
                else:
                    aggregated_aux_outputs[i].append(aux)

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

        # combine grad_chosen_inputs and grad_rejected_inputs

        grad_inputs = [torch.cat([grad_chosen_inputs, grad_rejected_inputs], dim=1) for
                       grad_chosen_inputs, grad_rejected_inputs in zip(grad_chosen_inputs, grad_rejected_inputs)]
        grad_inputs = torch.cat(grad_inputs, dim=1)
        policy_chosen_logps = torch.cat(policy_chosen_logps, dim=0)
        policy_rejected_logps = torch.cat(policy_rejected_logps, dim=0)

        # Aggregate aux outputs lists into tensors
        for i, aux in enumerate(aggregated_aux_outputs):
            if isinstance(aux, list):
                aggregated_aux_outputs[i] = torch.cat(aux, dim=0)

        ctx.save_for_backward(
            grad_inputs,
            grad_weight,
            grad_bias,
        )
        return_vars = (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits_mean,  # 只是显示，无法和 batch情况下对齐
            policy_rejected_logits_mean,
            policy_nll_loss,
        )
        return loss_acc, (*return_vars, *aggregated_aux_outputs)

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
            chosen_logits,
            rejected_logits,
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
        chosen_nll_loss = chosen_nll_loss / global_chosen_label_sum
        # TODO 是否要除以分母？ 只是显示
        chosen_logits_mean = chosen_logits.sum() / (chosen_bs * chosen_logits.shape[1] * weight.shape[0])
        rejected_logits_mean = rejected_logits.sum() / (
                chosen_bs * rejected_logits.shape[1] * weight.shape[0]
        )

        if use_ref_model:
            with torch.no_grad():
                (
                    ref_chosen_logps,
                    ref_rejected_logps,
                    ref_chosen_logits,
                    ref_rejected_logits,
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

        preference_loss_outputs = preference_loss_fn(
            chosen_logps, rejected_logps, chosen_bs, beta=beta, **loss_kwargs
        )
        if isinstance(preference_loss_outputs, tuple):
            preference_loss, *aux_outputs = preference_loss_outputs
        else:
            preference_loss, aux_outputs = preference_loss_outputs, []

        loss = alpha * chosen_nll_loss + preference_loss
        return_vars = (
            chosen_logps,
            rejected_logps,
            chosen_logits_mean,
            rejected_logits_mean,
            chosen_nll_loss,
        )
        return loss, (*return_vars, *aux_outputs)

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
        # TODO 这一步应该不对，dpo 并没有  / loss_mask.sum(-1)
        sum_logps = per_token_logps * loss_mask
        chosen_logps = sum_logps[:, :len_chosen_chunk].sum(-1) / loss_mask[:, :len_chosen_chunk].sum(-1)
        rejected_logps = sum_logps[:, len_chosen_chunk:].sum(-1) / loss_mask[:, len_chosen_chunk:].sum(-1)
        # average_log_prob = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        # chosen_logps = average_log_prob[:len_chosen_chunk]
        # rejected_logps = average_log_prob[len_chosen_chunk:]

        chosen_logits = logits_chunk[:, :len_chosen_chunk]
        rejected_logits = logits_chunk[:, len_chosen_chunk:]

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_nll_loss,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None, None, None


class LigerFusedLinearPackDPOLoss(torch.nn.Module):
    """
    Fused linear layer with DPO loss.
    """

    def __init__(
            self,
            ignore_index: int = -100,
            beta: float = 0.1,
            compute_nll_loss: bool = False,
            compiled: bool = True,
            use_ref_model: bool = False,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the odds ratio loss.
            compute_nll_loss (bool): Whether to compute the NLL loss.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.compute_nll_loss = compute_nll_loss
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
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
            num_tokens,
            global_chosen_label_sum
        )
