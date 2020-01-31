import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dir_result = './results/unet-bnorm/em/images'
lst_result = os.listdir(dir_result)

lst_input = [f for f in lst_result if f.endswith('input.png')]
lst_label = [f for f in lst_result if f.endswith('label.png')]
lst_output = [f for f in lst_result if f.endswith('output.png')]

nx = 512
ny = 512
nch = 1

n = 3
m = 5

inputs = torch.zeros((m, ny, nx, nch))
labels = torch.zeros((m, ny, nx, nch))
outputs = torch.zeros((m, ny, nx, nch))

for i in range(m):
    inputs[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_input[i]))[:, :, :nch])
    labels[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_label[i]))[:, :, :nch])
    outputs[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, lst_output[i]))[:, :, :nch])

inputs = inputs.permute((0, 3, 1, 2))
labels = labels.permute((0, 3, 1, 2))
outputs = outputs.permute((0, 3, 1, 2))
outputs = 1.0*(outputs > 0.5)

images = torch.cat([inputs, labels, outputs], axis=2)

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=False), (1, 2, 0)))

plt.show()

