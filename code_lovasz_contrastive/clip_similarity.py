import torch
import torchvision

# Download clip repo in clip files
from clip_files import model, clip

dataset = torchvision.datasets.CIFAR100(root='./', train=True)

text_labels = dataset.classes()
tokens = clip.tokenize(text_labels).to('cuda')

model, _ = clip.load('RN50', device='cuda')

text_features = model.encode_text(tokens)
text_features = text_features / text_features.norm(dim=1, keepdim=True)
similarity = text_features @ text_features.T

np.savetxt('similarity_graph_clipcsv', similarity.cpu().detach().numpy(), fmt='%.5f', delimiter=',')