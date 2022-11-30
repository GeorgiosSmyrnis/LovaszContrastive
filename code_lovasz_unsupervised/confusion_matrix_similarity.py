import torch
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix

def get_similarity(dataloader, encoder, classifier, opt):
    encoder.eval()
    classifier.eval()

    sim_mat = np.zeros((opt.n_cls, opt.n_cls))

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images = images.float().cuda()
            labels = labels.cuda()

            # forward
            output = classifier(encoder(images))
            preds = torch.argmax(output, dim=-1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            sim_mat = sim_mat + confusion_matrix(labels, preds, labels=np.arange(opt.n_cls), normalize=None)


        sim_mat = sim_mat / np.sum(sim_mat, axis=-1, keepdims=True)

        # Enforce symmetry
        sim_mat = (sim_mat + sim_mat.T)/2

        # Enforce complete self similarity
        for i in range(opt.n_cls):
            sim_mat[i,i] = 1

    return sim_mat
