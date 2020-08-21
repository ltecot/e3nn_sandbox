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

import glob
import re
import os

torch.set_default_dtype(torch.float64)

def extract_files_info():
    geom_dict = {}
    feat_dict = {}
    coef_dict = {}
    pattern = re.compile(".*_s_p_basis_density.out")
    for filename in glob.glob('./xyzfiles_1000/ts-stationary_*'):
        # print(filename)
        with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
            lines = [line.split() for line in f]
            key = [int(s) for s in filename.split('_') if s.isdigit()][-1]
            if pattern.match(filename):  # coef file
                # print("HEHERHEHRHHE1")
                data = []
                for a in range(6):
                    atom_coefs = []
                    for i in range(9):
                        atom_coefs.append(float(lines[2+11*a+i][1]))
                    data.append(atom_coefs)
                coef_dict[key] = torch.Tensor(data)
            else:  # geom file
                # print("HEHERHEHRHHE2")
                feats = []
                geom = []
                for a in range(2,8):
                    geom.append([float(lines[a][1]), float(lines[a][2]), float(lines[a][3])])
                    if lines[a][0] == 'O':
                        feats.append([1, 0])
                    elif lines[a][0] == 'H':
                        feats.append([0, 1])
                    else:
                        print("something is up")
                geom_dict[key] = torch.Tensor(geom)
                feat_dict[key] = torch.Tensor(feats)
    return geom_dict, feat_dict, coef_dict

geom_dict, feat_dict, coef_dict = extract_files_info()
# print(coef_dict)

Rs = [(2, 0)] # Two (2) scalar (L=0) channels: hydrogen and oxygen
Rs_out = [(3, 0), (2, 1)]

L_max = 2
model = GatedConvNetwork(Rs_in=Rs,
                         Rs_hidden=Rs_out,
                         Rs_out=Rs_out,
                         lmax=L_max,
                         max_radius=20.0,
                         kernel=Kernel)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss_fn = torch.nn.modules.loss.MSELoss()
# minibatch_size = 100
num_epochs = 5000

train_range = range(000, 900)
test_range = range(900, 1000)
for i in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    for j in train_range:
        output = model(feat_dict[j].unsqueeze(0), geom_dict[j].unsqueeze(0))
        loss = loss_fn(output, coef_dict[j])
        train_loss += loss
        # loss = (output - dummy_output).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 100 == 0:
            print("sample loss: ", loss)
    train_loss /= 900
    print("avg train loss: " + str(train_loss.item()))

    # Test
    model.eval()
    test_loss = 0
    for j in test_range:
        output = model(feat_dict[j].unsqueeze(0), geom_dict[j].unsqueeze(0))
        loss = loss_fn(output, coef_dict[j])
        test_loss += loss
    test_loss /= 100
    print("avg test loss: " + str(test_loss.item()))

