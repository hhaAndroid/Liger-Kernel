import torch
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss, LigerFusedLinearPackDPOLoss
from mmengine.runner import set_random_seed

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

input1 = batch_input_ids.detach().clone().requires_grad_(True)
pack_input_ids = pack_input_ids.detach().clone().requires_grad_(True)

# 要求输入的所有 chosen bs 都在前面，否则结果不对
loss1, aggregated_aux_outputs1 = liger_lm_head_dpo(input1, ref_batch_input_ids, batch_labels)
# print(loss1)
# print(aggregated_aux_outputs1)
loss_dict = {
            'dpo_losses': loss1,
            'chosen_rewards': aggregated_aux_outputs1[0].mean(),
            'rejected_rewards': aggregated_aux_outputs1[1].mean(),
            'reward_margin': ( aggregated_aux_outputs1[0].mean() - aggregated_aux_outputs1[1].mean()),
        }
print(loss_dict)

print('========================')
target_ = unpack_sequence(pack_labels, num_tokens)
global_chosen_label_sum = sum([(t != -100).sum() for t in target_[0::2]])
loss1, aggregated_aux_outputs1 = liger_lm_head_dpo(pack_input_ids, ref_pack_input_ids, pack_labels,
                                                   num_tokens=num_tokens, pack=True, global_chosen_label_sum=global_chosen_label_sum)
# print(loss1)
# print(aggregated_aux_outputs1)

loss_dict = {
            'dpo_losses': loss1,
            'chosen_rewards': aggregated_aux_outputs1[0].mean(),
            'rejected_rewards': aggregated_aux_outputs1[1].mean(),
            'reward_margin': (aggregated_aux_outputs1[0].mean() - aggregated_aux_outputs1[1].mean()),
        }
print(loss_dict)
