
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
# import ot
import torch.nn.functional as F
from AIDomains.abstract_layers import Sequential as absSequential
import torch.jit as jit

def margin_loss(logits, y, device):
    logit_org = logits.gather(1, y.view(-1, 1))
    y_target = (logits - torch.eye(10, device=device)[y] * 9999).argmax(1, keepdim=True)
    logit_target = logits.gather(1, y_target)
    loss = -logit_org + logit_target
    loss = loss.view(-1)
    return loss


class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma**(np.floor(self.current_step/self.interval))
        else:
            phase = len([x for x in self.interval if self.current_step>=x])
            return self.initial_step_size * self.gamma**(phase)

def adv_whitebox(model:absSequential, X:torch.Tensor, y:torch.Tensor, specLB:torch.Tensor, specUB:torch.Tensor, device, num_steps:int=200, step_size:float=0.2, lossFunc:str="margin", restarts:int=1, train:bool=True, teacher:absSequential=None):
    '''
    Conduct white-box adversarial attack on the model

    @param
        model: absSequential; the model to attack
        X: torch.Tensor; the batched input
        y: torch.Tensor; the batched label
        specLB: torch.Tensor; the lower bound of the input
        specUB: torch.Tensor; the upper bound of the input
        device: torch.device or str; the device to run the attack
        num_steps: int; the number of steps to run the attack
        step_size: float; the step size of the attack
        lossFunc: str; the loss function to use, should be one of ["pgd", "margin", "KL"]
        restarts: int; the number of restarts to run the attack
        train: bool; if True, will find the worst-case loss adversarial example; if False, will stop when adversarial example is found
        teacher: absSequential; teacher model for RSLAD style adversarial attacks

    @return
        adex: torch.Tensor; the adversarial example

    @remark
        An internal step size scheduler is used to adjust the step size during the attack.
        Adversarial accuracy is not returned and users should evaluate the returned adversarial examples by calling `model(adex)`.
    '''
    assert lossFunc in ["pgd", "margin", "KL"], "lossFunc should be either pgd, margin, or KL"
    assert (specUB-specLB).min() >= 0, "specLB cannot be smaller than specUB"
    assert (specUB >= X).all() and (X >= specLB).all(), "X should be within the bounds"
    adex = X.detach().clone()
    adex_found = torch.zeros(X.shape[0], dtype=bool, device=X.device)
    best_loss = torch.ones(X.shape[0], dtype=bool, device=X.device)*(-np.inf)
    lr_scale = (specUB-specLB)/2
    t_nat_probs = None

    if lossFunc == "KL" and teacher is not None:
        with torch.no_grad():
            t_outputs = teacher(X)
            t_nat_probs = F.log_softmax(t_outputs, dim=1)

    with torch.enable_grad():
        for _ in range(restarts):
            random_noise = torch.zeros_like(X).uniform_(-0.5, 0.5)*(specUB-specLB)
            X_pgd = (X + random_noise).clamp(specLB, specUB)

            if num_steps >= 50:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.5*num_steps), np.ceil(0.8*num_steps), np.ceil(0.9*num_steps)])
            elif num_steps >= 20:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.7*num_steps)])
            else:
                lr_scheduler = step_lr_scheduler(step_size, gamma=0.2, interval=10)

            for i in range(num_steps+1):
                X_pgd = X_pgd.detach()
                X_pgd.requires_grad = True
                if X_pgd.grad is not None:
                    X_pgd.grad.zero_()

                with torch.enable_grad():
                    out = model(X_pgd)
                    if lossFunc == 'pgd':
                        loss = nn.CrossEntropyLoss(reduction="none")(out, y)
                    elif lossFunc == "margin":
                        loss = margin_loss(out, y, device)
                    elif lossFunc == "KL" and teacher is None:
                        loss = nn.KLDivLoss(reduction='none')(F.log_softmax(out, dim=1), F.softmax(model(X), dim=1)).sum(dim=-1)
                    elif lossFunc == "KL" and teacher is not None:
                        s_adv_probs = F.log_softmax(out, dim=1)
                        loss = F.kl_div(s_adv_probs, t_nat_probs, log_target=True, reduction="none").sum(dim=-1)

                    is_adv = torch.argmax(out, dim=1) != y
                    if not train:
                        update_idx = is_adv
                    else:
                        update_idx = loss > best_loss
                    adex[update_idx] = X_pgd[update_idx].detach()
                    adex_found[update_idx] = True
                    best_loss[update_idx] = loss[update_idx].detach()

                    if (not train and adex_found.all()) and i==num_steps:
                        break

                if not train:
                    loss[adex_found] = 0.0

                loss = loss.sum()
                if torch.is_autocast_enabled():
                    # attack strength similar to without amp, tested on multiple model checkpoints
                    loss = loss * 2.**12 # no need to unscale it as we only need sign
                grad = torch.autograd.grad(loss, X_pgd)[0]
                assert not torch.isnan(grad).any(), "nan found in grad during attack; If automatic mixed precision is used, try a smaller scaling factor (usually not recommended). Otherwise, it usually indicates grad overflow due to inproper output scale."

                eta = lr_scheduler.get_lr() * lr_scale * grad.sign()
                lr_scheduler.step()
                X_pgd = (X_pgd + eta).clamp(specLB, specUB)

    assert (specUB >= adex).all() and (adex >= specLB).all(), "adex should be within the bounds"
    return adex