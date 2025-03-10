import torch
from affordance_clip import pixel_text_contrastive_loss

# read activation.pt and target.pt
activation = torch.load("activation.pt")
target = torch.load("target.pt")
target = target.float()

# print the shape of activation and target
print(activation.shape)
print(target.shape)


print(activation)
print(target)

# plot the activation and target and save to file
import matplotlib.pyplot as plt
plt.imshow(activation.cpu().squeeze(0).numpy())
plt.savefig("activation.png")
plt.imshow(target.cpu().squeeze(0).numpy())
plt.savefig("target.png")


# calculate the loss
loss = pixel_text_contrastive_loss(activation, target)
print(loss)

# calculate the min loss
min_loss = pixel_text_contrastive_loss(target, target)
print(min_loss)

# calculate the max loss
max_loss = pixel_text_contrastive_loss(activation, activation)
print(max_loss)