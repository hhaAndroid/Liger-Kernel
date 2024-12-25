import torch
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss, LigerFusedLinearPackDPOLoss
from mmengine.runner import set_random_seed
import torch.nn.functional as F

set_random_seed(42)


def unpack_sequence(packed: torch.Tensor,
                    num_tokens,
                    dim=1):
    if isinstance(num_tokens, torch.Tensor):
        num_tokens = num_tokens.tolist()
    sequences = torch.split(packed, num_tokens, dim=dim)
    return sequences


class LigerLMHeadDPO(torch.nn.Module):
    def __init__(
            self,
            H: int,
            V: int,
            dtype: torch.dtype,
            bias: bool = False,
            ref_bias: bool = False,
            compute_nll_loss: bool = False,
            ignore_index: int = -100,
            beta: float = 0.1,
            pack=False,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)

        self.dpo_loss_pack = LigerFusedLinearPackDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        )
        self.dpo_loss = LigerFusedLinearDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        )

    def forward(self, x, ref_x, y, pack=False, num_tokens=None, global_chosen_label_sum=None):
        if pack:
            return self.dpo_loss_pack(
                self.lin.weight,
                x,
                y,
                self.lin.bias,
                ref_x,
                self.ref_lin.weight,
                self.ref_lin.bias,
                num_tokens,
                global_chosen_label_sum,
            )
        else:
            return self.dpo_loss(
                self.lin.weight,
                x,
                y,
                self.lin.bias,
                ref_x,
                self.ref_lin.weight,
                self.ref_lin.bias,
            )


liger_lm_head_dpo = LigerLMHeadDPO(
    H=28,
    V=14,
    dtype=torch.bfloat16,
    bias=True,
    ref_bias=True,
    compute_nll_loss=True,
    ignore_index=-100,
    beta=0.1,
).cuda()

H = 28

a1 = torch.randn(5, H).bfloat16()
b1 = torch.Tensor([-100, -100, 1, 2, 3]).long()
a2 = torch.randn(8, H).bfloat16()
b2 = torch.Tensor([4, -100, 3, 4, 6, -100, -100, 7]).long()
a3 = torch.randn(3, H).bfloat16()
b3 = torch.Tensor([-100, 6, 8]).long()
a4 = torch.randn(4, H).bfloat16()
b4 = torch.Tensor([-100, 7, 8, -100]).long()
a5 = torch.randn(4, H).bfloat16()
b5 = torch.Tensor([-100, -100, 7, 4]).long()
a6 = torch.randn(3, H).bfloat16()
b6 = torch.Tensor([5, 8, -100]).long()

# 假设这个 label 已经是 shift 了的

max_item_length = 8
batch_input_ids = torch.zeros(8, max_item_length, H).cuda().bfloat16()
ref_batch_input_ids = torch.zeros(8, max_item_length, H).cuda().bfloat16()
batch_labels = torch.ones(8, max_item_length).long().cuda() * -100
ref_input_id = []
for i, (a, b) in enumerate([(a1, b1), (a2, b2), (a3, b3), (a2, b2), (a6, b6), (a4, b4), (a5, b5), (a6, b6)]):
    batch_input_ids[i, :a.size(0)] = a
    batch_labels[i, :b.size(0)] = b
    x = torch.randn(a.size(0), H).bfloat16()
    ref_batch_input_ids[i, :a.size(0)] = x
    ref_input_id.append(x)

# 特别注意：假设batch_input_ids中 bs 顺序是所有的 chosen bs 在前面，所有的 rejected bs 在后面
# 因此在 pack 组成数据时候要换顺序
num_tokens = torch.tensor([5, 3, 8, 4, 3, 4, 8, 3])
pack_input_ids = torch.cat([a1, a6, a2, a4, a3, a5, a2, a6], dim=0)[None, ...].cuda()
pack_labels = torch.cat([b1, b6, b2, b4, b3, b5, b2, b6], dim=0)[None, ...].cuda()
ref_input_id = [ref_input_id[i] for i in [0, 4, 1, 5, 2, 6, 3, 7]]
ref_pack_input_ids = torch.cat(ref_input_id, dim=0)[None, ...].cuda()

# 只用一条数据测试
# num_tokens = num_tokens[:2]
# pack_input_ids = pack_input_ids[:, :8]
# ref_pack_input_ids = ref_pack_input_ids[:, :8]
# pack_labels = pack_labels[:, :8]
# batch_input_ids = batch_input_ids[torch.tensor([0, 4])]
# ref_batch_input_ids = ref_batch_input_ids[torch.tensor([0, 4])]
# batch_labels = batch_labels[torch.tensor([0, 4])]

# 只用2条数据测试
num_tokens = num_tokens[:4]
pack_input_ids = pack_input_ids[:, :20]
ref_pack_input_ids = ref_pack_input_ids[:, :20]
pack_labels = pack_labels[:, :20]
batch_input_ids = batch_input_ids[torch.tensor([0, 1, 4, 5])]
ref_batch_input_ids = ref_batch_input_ids[torch.tensor([0, 1, 4,  5])]
batch_labels = batch_labels[torch.tensor([0, 1, 4, 5])]

input1 = batch_input_ids.detach().clone().requires_grad_(True)
# 要求输入的所有 chosen bs 都在前面，否则结果不对
loss1, aggregated_aux_outputs1 = liger_lm_head_dpo(input1, ref_batch_input_ids, batch_labels)
# print(loss1)
# print(aggregated_aux_outputs1)
loss_dict = {
    'dpo_losses': loss1,
    'chosen_rewards': aggregated_aux_outputs1[0].mean(),
    'rejected_rewards': aggregated_aux_outputs1[1].mean(),
    'reward_margin': (aggregated_aux_outputs1[0].mean() - aggregated_aux_outputs1[1].mean()),
}
print(loss_dict)
# loss1.backward()
# print(liger_lm_head_dpo.lin.weight.grad, liger_lm_head_dpo.lin.bias.grad)
# print(input1.grad)
# liger_lm_head_dpo.lin.weight.grad = None
# liger_lm_head_dpo.lin.bias.grad = None

pack_input_ids = pack_input_ids.detach().clone().requires_grad_(True)
print('========================')
target_ = unpack_sequence(pack_labels, num_tokens)
global_chosen_label_sum = sum([(t != -100).sum() for t in target_[0::2]])
loss1, aggregated_aux_outputs1 = liger_lm_head_dpo(pack_input_ids, ref_pack_input_ids, pack_labels,
                                                   num_tokens=num_tokens, pack=True,
                                                   global_chosen_label_sum=global_chosen_label_sum)
# print(loss1)
# print(aggregated_aux_outputs1)

loss_dict = {
    'dpo_losses': loss1,
    'chosen_rewards': aggregated_aux_outputs1[2],
    'rejected_rewards': aggregated_aux_outputs1[3],
    'reward_margin': aggregated_aux_outputs1[4],
    'policy_nll_loss': aggregated_aux_outputs1[6],
}
print(loss_dict)

# loss1.backward()
# print(liger_lm_head_dpo.lin.weight.grad, liger_lm_head_dpo.lin.bias.grad)
# print(pack_input_ids.grad)
# liger_lm_head_dpo.lin.weight.grad = None
# liger_lm_head_dpo.lin.bias.grad = None

import torch.nn as nn

def cross_entropy_loss(logits, labels):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels).sum()
    # loss = F.nll_loss(
    #     logits.view(-1, logits.shape[-1]),
    #     labels.view(-1),
    #     reduction="sum",
    #     ignore_index=-100,
    # )
    return loss


def get_pack_logps(
        all_logits_list,  # seqlen,vocab_size
        all_ref_logits_list,  # seqlen,vocab_size
        loss_mask_list,  # seqlen
):
    def compute_logps(_logps, _mask):
        _logps = _logps.sum(-1)
        # _logps = (_logps * _mask).sum(-1) / _mask.sum(-1)
        return _logps

    (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,
     reference_rejected_logps) = [], [], [], []
    for i in range(len(all_logits_list) // 2):
        chosen = all_logits_list[2 * i]
        rejected = all_logits_list[2 * i + 1]
        chosen_ref = all_ref_logits_list[2 * i]
        rejected_ref = all_ref_logits_list[2 * i + 1]
        chosen_mask = loss_mask_list[2 * i]
        rejected_mask = loss_mask_list[2 * i + 1]

        policy_chosen_logps.append(compute_logps(chosen, chosen_mask))
        policy_rejected_logps.append(
            compute_logps(rejected, rejected_mask))
        reference_chosen_logps.append(
            compute_logps(chosen_ref, chosen_mask))
        reference_rejected_logps.append(
            compute_logps(rejected_ref, rejected_mask))

    return (torch.stack(policy_chosen_logps),
            torch.stack(policy_rejected_logps),
            torch.stack(reference_chosen_logps),
            torch.stack(reference_rejected_logps))


pack_input_ids = pack_input_ids.detach().clone().requires_grad_(True)

# 对标 hf dpo 逻辑，看下是否一致
_labels = pack_labels.clone()
_labels[_labels == -100] = 0
loss_mask = _labels != 0
global_loss_mask = loss_mask

all_logits = liger_lm_head_dpo.lin(pack_input_ids).float()
with torch.no_grad():
    all_logits_ref = liger_lm_head_dpo.ref_lin(ref_pack_input_ids).float()

policy_logps = torch.gather(all_logits.log_softmax(-1), dim=2, index=_labels.unsqueeze(2)).squeeze(2)
ref_logps = torch.gather(all_logits_ref.log_softmax(-1), dim=2, index=_labels.unsqueeze(2)).squeeze(2)

policy_logps_list = unpack_sequence(policy_logps, num_tokens)
all_ref_logits_list = unpack_sequence(ref_logps, num_tokens)
loss_mask_list = unpack_sequence(global_loss_mask, num_tokens)
labels_list = unpack_sequence(pack_labels, num_tokens)
all_logits_list = unpack_sequence(all_logits, num_tokens)

shift_logits = torch.cat(all_logits_list[::2], dim=1)
shift_labels = torch.cat(labels_list[::2], dim=1)
# policy_nll_loss = cross_entropy_loss(shift_logits.float().log_softmax(-1), shift_labels)
policy_nll_loss = cross_entropy_loss(shift_logits, shift_labels)
policy_nll_loss = policy_nll_loss / global_chosen_label_sum

(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps) = get_pack_logps(
    policy_logps_list, all_ref_logits_list, loss_mask_list)

pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(0.1 * logits)
# loss = -F.logsigmoid(0.1 * logits).sum() / (len(num_tokens) // 2)
chosen_rewards = 0.1 * (policy_chosen_logps - reference_chosen_logps)
rejected_rewards = 0.1 * (policy_rejected_logps - reference_rejected_logps)
print('============================')
total_loss=(loss + policy_nll_loss).mean()
loss_dict = {
    'total_loss': total_loss.item(),
    'dpo_losses': loss.mean(),
    'policy_nll_loss': policy_nll_loss.mean(),
    'chosen_rewards': chosen_rewards.mean(),
    'rejected_rewards': rejected_rewards.mean(),
    'reward_margin': (chosen_rewards - rejected_rewards).mean(),
}
print(loss_dict)
# total_loss.backward()
# print(liger_lm_head_dpo.lin.weight.grad, liger_lm_head_dpo.lin.bias.grad)
# print(pack_input_ids.grad)
