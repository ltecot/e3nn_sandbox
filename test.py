import torch
import e3nn
import e3nn.point
import e3nn.radial
import e3nn.kernel
from e3nn.kernel_mod import Kernel
import e3nn.point.operations
from functools import partial
from e3nn.non_linearities import rescaled_act
from e3nn.networks import GatedConvNetwork
torch.set_default_dtype(torch.float64)

# 3D coordinates of the atoms of the molecule
C_geo = torch.tensor(
    [[ 0.     ,  1.40272,  0.     ],
    #  [-1.21479,  0.70136,  0.     ],
    #  [-1.21479, -0.70136,  0.     ],
    #  [ 0.     , -1.40272,  0.     ],
    #  [ 1.21479, -0.70136,  0.     ],
    #  [ 1.21479,  0.70136,  0.     ]
    ]
)
H_geo = torch.tensor(
    [[ 0.     ,  2.49029,  0.     ],
    #  [-2.15666,  1.24515,  0.     ],
    #  [-2.15666, -1.24515,  0.     ],
    #  [ 0.     , -2.49029,  0.     ],
    #  [ 2.15666, -1.24515,  0.     ],
    #  [ 2.15666,  1.24515,  0.     ]
    ]
)
geometry = torch.cat([C_geo, H_geo], axis=-2)

# and features on each atom
C_input = torch.tensor([[0., 1.] for i in range(C_geo.shape[-2])])
H_input = torch.tensor([[1., 0.] for i in range(H_geo.shape[-2])])
input = torch.cat([C_input, H_input])

dummy_output = torch.tensor(
    [[ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ],
     [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ],]
)

Rs = [(2, 0)] # Two (2) scalar (L=0) channels: hydrogen and carbon
Rs_out = [(3, 0), (2, 1)]

L_max = 2
# Rs = [(1, L) for L in range(L_max + 1)]
Network = partial(GatedConvNetwork,
                  Rs_in=Rs,
                  Rs_hidden=Rs_out,
                  Rs_out=Rs_out,
                  lmax=L_max,
                  max_radius=3.0,
                  kernel=Kernel)
model = Network()

## set up training

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# use 1e-3 if jumpy
optimizer.zero_grad()
loss_fn = torch.nn.modules.loss.MSELoss()

# minibatch_size = batch_size
num_epochs = 50000

for i in range(num_epochs):
    output = model(input.unsqueeze(0), geometry.unsqueeze(0))
    loss = loss_fn(output, dummy_output)
    # loss = (output - dummy_output).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(i, loss)

# # We are going to define RadialModel by specifying every single argument
# # of CosineBasisModel EXCEPT out_dim which will be passed later
# radial_layers = 2
# sp = rescaled_act.Softplus(beta=5)
# RadialModel = partial(e3nn.radial.CosineBasisModel, max_radius=max_radius,
#                       number_of_basis=number_of_basis, h=100,
#                       L=radial_layers, act=sp)

# K = partial(e3nn.kernel.Kernel, RadialModel=RadialModel)

# # If we wish to pass the convolution to a layer definition
# C = partial(e3nn.point.operations.Convolution, K)

# # Or alternatively, if we want to use the convolution directly,
# # we need to specify the `Rs` of the input and output
# Rs_in = [(2, 0)]
# Rs_out = [(3, 0), (2, 1)]
# convolution = e3nn.point.operations.Convolution(K(Rs_in, Rs_out))

# gated_block = nl.gated_block.GatedBlock(Rs_out, sp, rescaled_act.sigmoid)

# dimensionalities = [2 * L + 1 for mult, L in Rs_out for _ in range(mult)]
# gated_act = nl.GatedBlock(dimensionalities, rescaled_act.sigmoid, rescaled_act.sigmoid)
