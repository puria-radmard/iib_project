import torch
from LAS_systems.attention import SelfAttentionBase

# https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a

attention = SelfAttentionBase(4, 3, 3, 3)

input = torch.tensor(
    [[
        [1,0,1,0],
        [0,2,0,2],
        [1,1,1,1]
    ] for _ in range(50) ], dtype=float
).reshape(-1, 3, 4)

attention.Wk.weight = torch.nn.Parameter(
    torch.tensor(
        [[0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 0]],
        dtype=float
    ).T
)

attention.Wq.weight = torch.nn.Parameter(
    torch.tensor(
        [[1, 0, 1],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1]],
        dtype=float
    ).T
)

attention.Wv.weight = torch.nn.Parameter(
    torch.tensor(
        [[0, 2, 0],
        [0, 3, 0],
        [1, 0, 3],
        [1, 1, 0]],
        dtype=float
    ).T
)

attention.Wa.weight = torch.nn.Parameter(torch.eye(3, dtype=float))
attention.Wb.weight = torch.nn.Parameter(torch.eye(3, dtype=float))

output = attention(input)
