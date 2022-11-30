# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False,
                 method='moco', sim_mat=None, stability=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.method = method
        self.K = K
        self.m = m
        self.T = T

        self.stability = stability

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.method in ['supcon', 'lovasz', 'moco_lovasz', 'moco_supcon',
                           'moco_supcon2']:
            self.register_buffer("queue_labels", torch.ones(K).long() * -1)

        if sim_mat is None:
            self.sim_mat = sim_mat
        else:
            sim_mat = torch.from_numpy(np.loadtxt(sim_mat, delimiter=','))
            self.sim_mat = (sim_mat-sim_mat.min())/(sim_mat.max()-sim_mat.min())


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        if labels is not None: # replace the labels (AT SAME POINTER!)
            labels = concat_all_gather(labels)
            self.queue_labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr



    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, *args):
        if self.method == 'moco':
            return self.forward_moco(*args)
        elif self.method == 'supcon':
            return self.forward_supcon(*args)
        elif self.method == 'lovasz':
            return self.forward_lovasz(*args)
        elif self.method == 'moco_supcon':
            return self.forward_moco_sup(*args, lovasz=False)
        elif self.method == 'moco_lovasz':
            return self.forward_moco_sup(*args, lovasz=True)
        elif self.method == 'moco_supcon2':
            return self.forward_moco_sup2(*args)
        else:
            raise ValueError("Not valid method")

    # ======================================================================
    # =           Custom Supcon/Lovasz Methods                             =
    # ======================================================================

    def forward_supcon(self, im_q, im_k, labels):
        device = im_q.device
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k) # New keys: Nxc
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        self._dequeue_and_enqueue(k, labels)


        # Compute query features:
        q = self.encoder_q(im_q) # queries: Nxc
        q = nn.functional.normalize(q, dim=1)

        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # numerical_stability:
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        sims = torch.exp(logits / self.T)

        # Compute negatives
        neg_logits = torch.log(torch.sum(sims, dim=1))

        # Get mask and eval positives
        mask = labels.view(-1, 1).eq(self.queue_labels).float().to(device)
        pos_logits = torch.sum(logits * mask, dim=1) / torch.sum(mask, dim=1)


        return (-pos_logits + neg_logits).mean()


    def forward_lovasz(self, im_q, im_k, labels):
        device = im_q.device
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k) # New keys: Nxc
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        self._dequeue_and_enqueue(k, labels)


        # Compute query features:
        q = self.encoder_q(im_q) # queries: Nxc
        q = nn.functional.normalize(q, dim=1)
        mask = labels.view(-1, 1).eq(self.queue_labels).float().to(device)

        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        similarities = (self.sim_mat[labels, :][:, self.queue_labels]).to(device)

        # Compute negatives
        neg_logits = (logits - similarities) / (1 - similarities + 1e-6)
        neg_logits = neg_logits * (1 - mask) # STABILITY HACK
        pos_logits = logits

        # Interject to add numerical stability if flagged
        if self.stability:
            max_logits = torch.max(neg_logits, dim=1, keepdim=True)[0].detach()
            neg_logts = neg_logits - max_logits
            pos_logits = pos_logits - max_logits


        neg_logits = torch.exp(neg_logits / self.T)
        neg_logits = neg_logits * (1 - mask) # STABILITY HACK
        neg_logits = torch.log(neg_logits.sum(dim=1))
        # Get mask and eval positives

        pos_logits = torch.sum(pos_logits * mask / self.T, dim=1) / torch.sum(mask, dim=1)

        return (-pos_logits + neg_logits).mean()


    # ============================================================================
    # =           MOCO-ADAPTED Supervised Methods                                =
    # ============================================================================

    def forward_moco_sup2(self, im_q, im_k, labels):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)


        self._dequeue_and_enqueue(k, labels)
        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        mask = torch.eq(labels.view(-1, 1), self.queue_labels)

        loss_sup = (-torch.log_softmax(logits, dim=-1) * mask).sum(
            dim=-1, keepdim=True).div(mask.sum(dim=-1, keepdim=True) + 1e-5)
        loss_sup = loss_sup.mean()
        return loss_sup



    def forward_moco_sup(self, im_q, im_k, labels, lovasz=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # Compute all Query<->Key dot products
        mask = labels.view(-1, 1).eq(self.queue_labels).float().to(q.device)
        logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # And then compute positive logits
        # (pulling positive bank from key AND queue)
        pos_sim = torch.einsum('nc,nc->n', [q, k])
        l_pos = ((logits * mask).sum(dim=1) +  pos_sim)/ (mask.sum(dim=1) + 1)

        # Now compute negative logits
        l_neg = logits
        if lovasz: # If Lovasz, update with similarities
            similarities = (self.sim_mat[labels, :][:, self.queue_labels]).to(q.device)
            similarities[mask == 1] = 0.0 # these get masked out later...
            assert similarities.shape == l_neg.shape
            l_neg = (l_neg - similarities) / torch.clamp(1 - similarities, min=1e-6, max=1.0)

        l_neg = l_neg * (1 - mask)


        # logits: Nx(1+K)
        all_logits = torch.cat([l_pos.view(-1, 1), l_neg], dim=1)

        # apply temperature
        all_logits /= self.T

        # targets: positive key indicators
        targets = torch.zeros(all_logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)

        return all_logits, targets


    # ======================================================================
    # =           Regular MoCo                                             =
    # ======================================================================

    def forward_moco(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
