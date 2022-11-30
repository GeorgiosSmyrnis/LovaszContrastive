from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, pos_reg_base=1.0, neg_reg_base=1.0, sim_mat=None, stability=True, blend=1.0, pair=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.pos_reg_base = pos_reg_base
        self.neg_reg_base = neg_reg_base
        self.pair = pair
        #self.sim_mat = sim_mat.cuda() if sim_mat is not None else None
        if blend is None:
            blend = 1.0
        self.sim_mat = (blend * sim_mat.cuda() + (1-blend) * torch.eye(sim_mat.shape[0]).cuda()) if sim_mat is not None else None
        self.stability = stability
        self.blend = blend

    def forward(self, features, labels=None, sample_sim_mat=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        sample_sim_mat: The sample similarity matrix used for unsupervised Lovasz contrastive learning.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = features / torch.linalg.norm(features, dim=-1, keepdim=True)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        logits = anchor_dot_contrast
       

        # tile mask
        orig_mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(orig_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = orig_mask * logits_mask

        if self.sim_mat is None and sample_sim_mat is None:
            # for numerical stability
            logits = torch.div(logits, self.temperature)
            logits_max, _ = torch.max(logits, dim=1, keepdim=True)
            logits = logits - logits_max.detach()

            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

        else:
            logits_pos = torch.div(logits, self.temperature) * logits_mask * orig_mask
            if sample_sim_mat is None:
                temp_labels = torch.cat([labels, labels], dim=0).flatten()
                similarities = self.sim_mat[temp_labels,:][:,temp_labels]
            else:
                similarities = self.blend * sample_sim_mat + (1-self.blend) * orig_mask  #mask is essentially the similarity matrix for SimCLR
            logits_neg = torch.full_like(logits, 0)
            sim_mask = (similarities < 1)
            logits_neg[sim_mask] = (logits[sim_mask] - similarities[sim_mask])/(1-similarities[sim_mask])
            logits_neg = torch.div(logits_neg, self.temperature) * logits_mask * (1-orig_mask)
            if self.stability:
                max_logits = torch.max(logits_neg, dim=1, keepdim=True)[0].detach()
                logits_neg = logits_neg - max_logits
                logits_pos = logits_pos - max_logits

            exp_logits_neg = torch.exp(logits_neg) * logits_mask * (1-orig_mask)
            loss_pos = -(logits_pos.sum(dim = 1))/(logits_mask * orig_mask).sum(dim=1)
            loss_neg = torch.log(exp_logits_neg.sum(dim=1))
            if self.pair:
                return loss_pos.mean(), loss_neg.mean()

            loss = loss_pos.mean() + loss_neg.mean()
            loss = (self.temperature / self.base_temperature) * loss
        
        return loss
