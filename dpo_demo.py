import torch
from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss


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
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.dpo_loss = LigerFusedLinearDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        )

    def forward(self, x, ref_x, y):
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


B = 4
T = 5
H = 28

_input = torch.randn(B, T, H, device='cuda', dtype=torch.bfloat16) * 10
input1 = _input.detach().clone().requires_grad_(True)

ref_input = torch.randn(B, T, H, device='cuda', dtype=torch.bfloat16, requires_grad=False) * 10

target = torch.randint(
        0,
        14,
        (
            B,
            T,
        ),
        device='cuda',
        dtype=torch.long,
    )

num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
target.view(-1)[indices_to_assign] = -100

loss1, aggregated_aux_outputs1 = liger_lm_head_dpo(input1, ref_input, target)
print(loss1)
print(aggregated_aux_outputs1)
