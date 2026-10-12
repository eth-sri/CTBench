import copy
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from attacks import adv_whitebox

from AIDomains.zonotope import HybridZonotope
import AIDomains.abstract_layers as abs_layers
from AIDomains.abstract_layers import AbstractModule, Sequential as absSequential
import AIDomains.concrete_layers as conc_layers
from AIDomains.wrapper import propagate_abs
from AIDomains.ai_util import construct_C
from typing import Callable, Iterable, List, Tuple, Final, Union, Optional
from utils import project_to_bounds, log_cuda_memory, seed_everything
import logging
import torch.jit as jit

def get_model_wrapper(args, net, device, input_dim):
    # Define model wrapper here
    if args.use_pgd_training:
        model_wrapper = PGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=args.use_EDAC_step)
    elif args.use_multipgd_training:
        model_wrapper = MultiPGDModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, enable_EDAC=args.use_EDAC_step)
    elif args.use_arow_training:
        model_wrapper = ARoWModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.arow_reg_weight, args.arow_label_smoothing, enable_EDAC=args.use_EDAC_step)
    elif args.use_mart_training:
        model_wrapper = MARTModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.mart_reg_weight, enable_EDAC=args.use_EDAC_step)
    elif args.use_ibp_training:
        if args.use_small_box:
            model_wrapper = SmallBoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, eps_shrinkage=args.eps_shrinkage, relu_shrinkage=args.relu_shrinkage)
        else:
            model_wrapper = BoxModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
    elif args.use_taps_training:
        if args.use_small_box:
            model_wrapper = STAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=args.block_sizes, eps_shrinkage=args.eps_shrinkage, relu_shrinkage=args.relu_shrinkage)
        else:
            model_wrapper = TAPSModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, block_sizes=args.block_sizes, relu_shrinkage=args.relu_shrinkage)
    elif args.use_DP_training:
        model_wrapper = DeepPolyModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, use_dp_box=False)
    elif args.use_DPBox_training:
        model_wrapper = DeepPolyModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, use_dp_box=True, loss_fusion=args.use_loss_fusion, keep_fusion_when_test=args.keep_fusion_when_test)
    elif args.use_mtlibp_training:
        model_wrapper = MTLIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.ibp_coef, args.attack_range_scale)
    elif args.use_expibp_training:
        model_wrapper = EXPIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.ibp_coef, args.attack_range_scale)
    elif args.use_ccibp_training:
        model_wrapper = CCIBPModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.ibp_coef, args.attack_range_scale)
    elif args.use_ccdist_training:
        model_wrapper = CCDistModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.ibp_coef, args.attack_range_scale)
    elif args.use_adcert_training:
        model_wrapper = ADCERTModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args, args.ibp_coef, args.attack_range_scale)
    elif args.use_std_training:
        model_wrapper = BasicModelWrapper(net, nn.CrossEntropyLoss(), input_dim, device, args)
    else:
        raise NotImplementedError("Unknown training mode.")
    if args.grad_accu_batch is not None:
        model_wrapper = GradAccuFunctionWrapper(model_wrapper, args)
    if args.use_sam:
        model_wrapper = SAMFunctionWrapper(model_wrapper, args.sam_rho, args.adaptive_sam_rho)
    if args.use_weight_smooth:
        model_wrapper = WeightSmoothFunctionWrapper(model_wrapper, std_scale=args.weight_smooth_std_scale)
    return model_wrapper

def set_value_between(self, name:str, value, lower, upper, dtype:Callable):
    '''Value checker for properties. Checks whether the value is between lower and upper.'''
    value = dtype(value)
    assert lower <= value <= upper, f"{name} should be between {lower} and {upper}."
    setattr(self, name, value)

def set_value_typecheck(self, name:str, value, dtype):
    '''Value checker for properties. Checks whether the value is of the specified type.'''
    assert isinstance(value, dtype), f"{name} should be of type {dtype}."
    setattr(self, name, value)

def set_value_typecast(self, name:str, value, dtype:Callable, constraint:Optional[Callable]=None, msg:Optional[str]=None):
    '''Value checker for properties. Casts the value to the specified type.'''
    if constraint:
        assert constraint(value), msg
    setattr(self, name, dtype(value))

class BasicModelWrapper(nn.Module):
    '''
    Implements standard training procedure.

    @param
        net: Callable; the model to be trained
        loss_fn: Callable; the loss function
        input_dim: Tuple[int, ...]; the input dimension of the model
        device: the device to run the model, e.g., "cuda" or "cpu"
        args: the arguments from the argument parser
        data_range: Tuple[float, float]; the range of the input data, e.g., (0, 1) for images. Default is (0, 1)

    @property
        robust_weight: float; the weight of the robust loss in the combined loss. Default is None, i.e., calling combine_loss will raise an error.
        summary_accu_stat: bool; whether to return the accuracy statistics in the summary. Default is True.
        grad_cleaner: read-only; the optimizer to clean the grad; used to call zero_grad only
        data_min: float; the minimum value of the input data. Default is 0.0.
        data_max: float; the maximum value of the input data. Default is 1.0.
        freeze_BN: bool; whether to freeze the BN stat during training. Default is False.
        current_eps: float; the current perturbation radius. Default is 0.0.
        device: the device to run the model, e.g., "cuda" or "cpu"; can only be modified by wrapper.to(device)
    '''
    robust_weight = property(fget=lambda self: self._robust_weight, fset=lambda self, value: set_value_between(self, "_robust_weight", value, 0, 1, float))
    summary_accu_stat = property(fget=lambda self: self._summary_accu_stat, fset=lambda self, value: set_value_typecheck(self, "_summary_accu_stat", value, bool))
    grad_cleaner = property(fget=lambda self: self._grad_cleaner, fset=None) # read-only
    data_min = property(fget=lambda self: self._data_min, fset=lambda self, value: set_value_typecast(self, "_data_min", value, float, lambda x: x <= self.data_max, "data_min should be less than or equal to data_max"))
    data_max = property(fget=lambda self: self._data_max, fset=lambda self, value: set_value_typecast(self, "_data_max", value, float, lambda x: x >= self.data_min, "data_max should be greater than or equal to data_min"))
    freeze_BN = property(fget=lambda self: self._freeze_BN, fset=lambda self, value: set_value_typecheck(self, "_freeze_BN", value, bool))
    current_eps = property(fget=lambda self: self._current_eps, fset=lambda self, value: set_value_typecast(self, "_current_eps", value, float, lambda x: x>=0 and x<=self.max_eps, "max_eps needs to be set before current_eps, and current_eps must be a non-negative float smaller than or equal to max_eps."))
    current_lr = property(fget=lambda self: self._current_lr, fset=lambda self, value: set_value_typecast(self, "_current_lr", value, float, lambda x: x>0, "current_lr must be a positive float."))
    grad_scaler = property(fget=lambda self: self._grad_scaler, fset=None) # read-only
    max_eps = property(fget=lambda self: self._max_eps, fset=lambda self, value: set_value_typecast(self, "_max_eps", value, float, lambda x: x>=self.current_eps, "max_eps must be a float larger than current_eps."))
    device = property(fget=lambda self: self._device, fset=None)

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, data_range:Tuple[float, float]=(0.0, 1.0)):
        super().__init__()
        self.net = net
        self.BNs = [layer for layer in self.net if isinstance(layer, abs_layers._BatchNorm)]
        self._freeze_BN:bool = False
        self.loss_fn = loss_fn
        self.args = args
        self.input_dim = input_dim # currently not used by any since the net already sets the dimensions
        self._device = device
        self._summary_accu_stat:bool = True
        self._data_min:float = float(data_range[0])
        self._data_max:float = float(data_range[1])
        assert self._data_min <= self._data_max, "data_min should be less than or equal to data_max."
        self._robust_weight = None
        self._current_eps = 0.
        self._current_lr = None
        self._grad_cleaner:Final = torch.optim.SGD(self.net.parameters(), lr=1) # will only call zero_grad on it
        self._max_eps = max(args.train_eps if hasattr(args, "train_eps") and args.train_eps is not None else 0, args.test_eps if hasattr(args, "test_eps") and args.test_eps is not None else 0)
        self.use_amp = hasattr(args, "use_amp") and args.use_amp
        self._grad_scaler:Final = torch.cuda.amp.GradScaler() if self.use_amp else None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def to(self, device):
        self._device = device
        self.net = self.net.to(device)
        if hasattr(self, "teacher"):
            self.teacher = self.teacher.to(device)
        if hasattr(self, "wrapper"):
            self.wrapper = self.wrapper.to(device)
        return self

    def Get_Performance(self, x:torch.Tensor, y:torch.Tensor, use_model:Optional[Callable]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        '''
        Compute standard statistics from the clean input.

        @param
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label
            use_model: Callable; the model to be used; if None, use the model in the wrapper, i.e., self.net

        @return
            loss: torch.Tensor; the loss
            accu: torch.Tensor; the accuracy
            pred_correct: torch.BoolTensor; whether the prediction for each input in the batch is correct
        '''
        if use_model is None:
            outputs = self.net(x)
        else:
            outputs = use_model(x)
        loss = self.loss_fn(outputs, y)
        # detach the loss to save memory when natural loss is not used for training
        if self.robust_weight == 1:
            loss = loss.detach()
        accu, pred_correct = self._Get_Accuracy(outputs, y)
        return loss, accu, pred_correct

    def _Get_Accuracy(self, outputs:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        '''
        Compute the accuracy from the prediction and the label. Not supposed to be called by clients.

        @param
            outputs: torch.Tensor; the batched prediction
            y: torch.Tensor; the batched label

        @return
            accu: torch.Tensor; the accuracy
            pred_correct: torch.BoolTensor; whether the prediction for each input in the batch is correct
        '''
        assert len(outputs) == len(y), 'prediction and label should match.'
        pred_correct = torch.argmax(outputs, dim=1).eq(y)
        num_correct = torch.sum(pred_correct)
        return num_correct / len(y), pred_correct

    def _set_BN(self, BN_layers:List[abs_layers._BatchNorm], update_stat:Optional[bool]=None):
        '''
        Set the update_stat attribute of the BN layers.

        @param
            BN_layers: List[abs_layers._BatchNorm]; the list of BN layers
            update_stat: bool; whether to update the BN stat when training=True; if None, will use the existing BN stat instead
        '''
        if update_stat is not None:
            for layer in BN_layers:
                layer.update_stat = update_stat

    def compute_nat_loss_and_set_BN(self, x:torch.Tensor, y:torch.Tensor, **kwargs):
        '''
        @param
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label
            (optional kwargs) use_model: Callable; the model to be used; by default, use the model in the wrapper, i.e., self.net

        @return
            nat_loss: torch.Tensor; the natural loss
            nat_accu: torch.Tensor; the natural accuracy
            is_nat_accu: torch.BoolTensor; whether the prediction for each input in the batch is correct

        @remark
            Batch norm stat will not be updated if self.freeze_BN is True.
        '''
        assert (x>=self.data_min).all() and (x<=self.data_max).all(), "Input data should be within the data range."
        self._set_BN(self.BNs, update_stat=not self.freeze_BN)
        result = self.Get_Performance(x, y, **kwargs)
        self._set_BN(self.BNs, update_stat=False)
        return result
    
    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        '''
        Compute the robust loss and accuracy from the bounds of the input data.

        @param
            lb: torch.Tensor; the lower bound of the input data
            ub: torch.Tensor; the upper bound of the input data
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label

        @return
            robust_loss: torch.Tensor; the robust loss
            robust_accu: torch.Tensor; the robust accuracy
            is_robust_accu: torch.BoolTensor; whether the prediction for each input in the batch is robustly correct
        '''
        raise NotImplementedError
    
    def get_robust_stat_from_input_noise(self, eps:float, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        '''
        Compute the robust loss and accuracy from the input noise, i.e., the perturbation radius eps projected into the given data range [self.data_min, self.data_max].

        @param
            eps: float; the perturbation radius
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label

        @return
            robust_loss: torch.Tensor; the robust loss
            robust_accu: torch.Tensor; the robust accuracy
            is_robust_accu: torch.BoolTensor; whether the prediction for each input in the batch is robustly correct

        @remark
            The input data x should be within the data range [self.data_min, self.data_max].
            This will internally call get_robust_stat_from_bounds with the projected bounds.
        '''
        return self.get_robust_stat_from_bounds((x - eps).clamp(min=self.data_min), (x + eps).clamp(max=self.data_max), x, y)

    def combine_loss(self, nat_loss:torch.Tensor, robust_loss:torch.Tensor) -> torch.Tensor:
        '''
        return (1 - self.robust_weight) * nat_loss + self.robust_weight * robust_loss

        @param
            nat_loss: torch.Tensor; the natural loss
            robust_loss: torch.Tensor; the robust loss

        @return
            loss: torch.Tensor; the combined loss

        @remark
            self.robust_weight should be set before calling combine_loss.
        '''
        assert self.robust_weight is not None, "robust_weight should be set before calling combine_loss."
        loss = (1 - self.robust_weight) * nat_loss + self.robust_weight * robust_loss
        return loss
    
    def grad_postprocess(self) -> None:
        '''
        A wrapper function to be called right before optimizer.step(); can be used to modify the grad.

        @remark
            By default, it clips the grad norm. This can be overriden to implement any gradient processing.
        '''
        # grad clipping
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_clip)

    def param_postprocess(self) -> None:
        '''
        A wrapper function to be called right after optimizer.step(); can be used to modify the parameters.

        @remark
            By default, it does nothing. This can be overriden to implement any parameter processing.
            For example, when gradients are computed with shiftied parameters, the parameters can be shifted back here. For example, see SAMFunctionWrapper.
        '''
        pass


    def format_return(self, loss:torch.Tensor, nat_loss:torch.Tensor, nat_accu:torch.Tensor, is_nat_accu:torch.BoolTensor, robust_loss:torch.Tensor, robust_accu:torch.Tensor, is_robust_accu:torch.BoolTensor):
        '''
        Format the return values based on the self.summary_accu_stat.

        @param
            loss: torch.Tensor; the combined loss
            nat_loss: torch.Tensor; the natural loss
            nat_accu: torch.Tensor; the natural accuracy
            is_nat_accu: torch.BoolTensor; whether the prediction for each input in the batch is correct
            robust_loss: torch.Tensor; the robust loss
            robust_accu: torch.Tensor; the robust accuracy
            is_robust_accu: torch.BoolTensor; whether the prediction for each input in the batch is robustly correct

        @remark
            This is usually the last function to be called in the compute_model_stat method to format the return values.
        '''
        if self.summary_accu_stat:
            return (loss, nat_loss, robust_loss), (nat_accu.item(), robust_accu.item())
        else:
            return (loss, nat_loss, robust_loss), (nat_accu.item(), robust_accu.item()), (is_nat_accu, is_robust_accu)
        
    def compute_model_stat(self, x:torch.Tensor, y:torch.Tensor, eps:float, **kwargs):
        '''
        The main function to compute the loss and accuracy statistics. Subclasses should override this function to implement the specific training procedure.

        @param
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label
            eps: float; the perturbation radius

        @return
            the return values formatted by self.format_return

        @remark
            Subclasses should call self.compute_nat_loss_and_set_BN and self.get_robust_stat_from_input_noise to compute the natural loss and robust loss, respectively. This method is responsible for updating self.current_eps.
            When self.freeze_BN is True, the BN stat will not be updated during training.
            If self.get_robust_stat_from_bounds is not implemented, then this will check whether eps is 0. If eps is zero, then the robust loss is set to 0 and the robust accuracy is set to natural accuracy. Otherwise, it raises a NotImplementedError.
        '''
        self.current_eps = eps
        nat_loss, nat_accu, is_nat_accu = self.compute_nat_loss_and_set_BN(x, y)
        try:
            robust_loss, robust_accu, is_robust_accu = self.get_robust_stat_from_input_noise(eps, x, y)
            loss = self.combine_loss(nat_loss, robust_loss)
        except NotImplementedError:
            if eps == 0:
                loss, robust_loss, robust_accu, is_robust_accu = nat_loss, torch.zeros_like(nat_accu), nat_accu, is_nat_accu
            else:
                raise NotImplementedError("Robust computation is not implemented.")
        return self.format_return(loss, nat_loss, nat_accu, is_nat_accu, robust_loss, robust_accu, is_robust_accu)
        
    def convert_pred_to_margin(self, y:torch.Tensor, pred:torch.Tensor, rearrange_label:bool=True) -> torch.Tensor:
        '''
        Convert the prediction to the margin, i.e., the difference between the prediction and the target class, i.e. pred_i - pred_{y}. If rearrange_label is True, then the true class will be rearranged to the first column, i.e., loss(pred, y) == loss(margin, 0). Otherwise, loss(pred, y) == loss(margin, y).

        @param
            y: torch.Tensor; the batched label
            pred: torch.Tensor; the batched prediction
            rearrange_label: bool; whether to rearrange the label to the first column

        @return
            margin: torch.Tensor; the batched margin

        @remark
            The margin is padded with 0 at the first column by definition when rearrange_label=True. This implementation is based on dense matrix product and thus might be inefficient when the number of classes is large. However, it is usually not a practical issue in certified training.
        '''
        assert len(y) == len(pred), "y and pred should have the same batch size."
        if rearrange_label:
            C = construct_C(self.net.output_dim[-1], y)
            margin = - torch.bmm(C, pred.unsqueeze(-1)).squeeze(-1)
            margin = torch.cat((torch.zeros(size=(margin.size(0), 1), dtype=margin.dtype, device=margin.device), margin), dim=1)
            # postcondition: margin[i, 0] == 0 for 0<=i<len(y)
        else:
            margin = pred - pred[torch.arange(len(y)), y].unsqueeze(-1)
            # postcondition: margin[i, y[i]] == 0 for 0<=i<len(y)
        return margin

class _EDACMixin:
    '''
    Implements Extragradient-type method to explicitly Decrease Adversarial Certainty (EDAC).
    EDAC is a method to reduce the adversarial certainty of the model to avoid adversarial overfitting.
    Reference: https://arxiv.org/abs/2310.04539; https://github.com/TrustMLRG/EDAC

    @property
        EDAC_optimizer: torch.optim.Optimizer; the optimizer for the EDAC step
        EDAC_step_size: float; the step size for the EDAC step

    @remark
        Subclasses should call EDAC_step in the compute_model_stat method to perform the EDAC step.
    '''
    EDAC_step_size = property(fget=lambda self: self._EDAC_step_size, fset=lambda self, value: set_value_typecast(self, "_EDAC_step_size", value, float))
    EDAC_optimizer:Final = property(fget=lambda self: self._EDAC_optimizer, fset=None) # read-only

    def register_EDAC_hyperparam(self, optimizer, EDAC_step_size:float=0.3) -> None:
        self._EDAC_optimizer = optimizer
        self._EDAC_step_size = float(EDAC_step_size)

    def EDAC_step(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> None:
        '''
        Perform the EDAC step as described in the paper.

        @param
            lb: torch.Tensor; the lower bound of the input data
            ub: torch.Tensor; the upper bound of the input data
            x: torch.Tensor; the batched input data
            y: torch.Tensor; the batched label

        @remark
            This should be called in the compute_model_stat method to do a pre-update of the model.
        '''
        assert hasattr(self, "EDAC_optimizer") and hasattr(self, "EDAC_step_size"), "EDAC hyperparam not registered. Call register_EDAC_hyperparam(optimizer, EDAC_step_size) first."
        xadv = adv_whitebox(self.net, x, y, lb, ub, self.device, self.num_steps, step_size=max(0.25, 2/self.num_steps), restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        robust_std = torch.std(yadv, dim=1)
        advCertainty = robust_std.mean()
        ac_loss = self.EDAC_step_size * advCertainty
        self.EDAC_optimizer.zero_grad()
        with torch.autocast(device_type=self.device, enabled=False):
            if self.use_amp:
                self.grad_scaler.scale(ac_loss).backward()
                self.grad_scaler.step(self.EDAC_optimizer)
                self.grad_scaler.update()
            else:
                ac_loss.backward()
                self.EDAC_optimizer.step()

def cache_adv_xy_if_need(self, advx:Optional[torch.Tensor]=None, advy:Optional[torch.Tensor]=None) -> None:
    '''
    Cache the adversarial examples and the model's output on the adversarial examples if cache_advx=True and cache_advy=True, respectively. When False, the cache will be cleared.

    @param
        advx: torch.Tensor; the adversarial examples
        advy: torch.Tensor; the model's output on the adversarial examples

    @remark
        advy will be inferred from advx if advy is None and advx is provided when cache_advy=True.
        As a utility function, we do not write this as part of the model wrapper but a separate function to avoid nested design for non-adversarial wrappers, similar to the adv_whitebox function.
    '''
    assert hasattr(self, "cache_advx") and hasattr(self, "cache_advy"), "cache_advx and cache_advy should be defined in the model wrapper in order to call cache_adv_xy_if_need."
    with torch.no_grad():
        if self.cache_advx:
            assert isinstance(advx, torch.Tensor), "advx should be a torch.Tensor to be cached."
            self.current_advx = advx.clone().detach()
        else:
            self.current_advx = None
        if self.cache_advy:
            if advy is None:
                assert advx is not None, "advy cannot be inferred from advx if advx is not cached."
                advy = self.net(advx)
            assert isinstance(advy, torch.Tensor), "advy should be a torch.Tensor to be cached."
            self.current_advy = advy.clone().detach()
        else:
            self.current_advy = None

class PGDModelWrapper(_EDACMixin, BasicModelWrapper):
    '''
    Implements PGD training.

    @param
        net: Callable; the model to be trained
        loss_fn: Callable; the loss function
        input_dim: Tuple[int, ...]; the input dimension of the model
        device: the device to run the model, e.g., "cuda" or "cpu"
        args: the arguments from the argument parser
        enable_EDAC: bool; whether to enable EDAC step. If True, then register_EDAC_hyperparam should be called to register the optimizer and step size. By default, False.
        cache_advx: bool; whether to cache the adversarial examples found by PGD. By default, False.
        cache_advy: bool; whether to cache the model's output on the adversarial examples. By default, False.

    @property
        num_steps: int; the number of steps for PGD. By default, it is set to args.train_steps.
        restarts: int; the number of restarts for PGD. By default, it is set to args.restarts.
        current_advx: torch.Tensor; the current adversarial examples
        current_advy: torch.Tensor; the current model's output on the adversarial examples
    
    @remark
        If subclasses would like to use EDAC step and overrides compute_model_stat, make sure to copy the EDAC step code from compute_model_stat. If compute_model_stat is not overridden, then EDAC step will be inherited automatically and can be enabled via setting enable_EDAC=True and calling register_EDAC_hyperparam.

        Cache of adversarial examples and model's output on adversarial examples is useful when the same adversarial examples are used for multiple purposes, e.g., adversarial training and adversarial evaluation. However, note that the cache will consume extra memory and should be cleared when not needed. Cache of new adversarial examples will overwrite the old ones.
    '''
    num_steps = property(fget=lambda self: self._num_steps, fset=lambda self, value: set_value_typecast(self, "_num_steps", value, int, lambda x: x>0, "num_steps must be a positive integer."))
    restarts = property(fget=lambda self: self._restarts, fset=lambda self, value: set_value_typecast(self, "_restarts", value, int, lambda x: x>0, "restarts must be a positive integer."))
    enable_EDAC = property(fget=lambda self: self._enable_EDAC, fset=lambda self, value: set_value_typecheck(self, "_enable_EDAC", value, bool))
    cache_advx = property(fget=lambda self: self._cache_advx, fset=lambda self, value: set_value_typecheck(self, "_cache_advx", value, bool))
    cache_advy = property(fget=lambda self: self._cache_advy, fset=lambda self, value: set_value_typecheck(self, "_cache_advy", value, bool))
    step_size = property(fget=lambda self: self._step_size, fset=lambda self, value: set_value_typecast(self, "_step_size", value, float, lambda x: x>0, "step_size must be a positive float."))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, enable_EDAC:bool=False, cache_advx:bool=False, cache_advy:bool=False, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args)
        # change robust_weight directly during steps instead of modifying args
        self._num_steps = int(args.train_steps)
        self._restarts = int(args.restarts)
        self._enable_EDAC = bool(enable_EDAC)
        self._cache_advx = bool(cache_advx)
        self._cache_advy = bool(cache_advy)
        self.current_advx = None
        self.current_advy = None
        self._step_size = max(0.25, 2/self.num_steps) if args.step_size is None else args.step_size

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        xadv = adv_whitebox(self.net, x, y, lb, ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        adv_accu, is_adv_accu = self._Get_Accuracy(yadv, y)
        adv_loss = self.loss_fn(yadv, y)
        cache_adv_xy_if_need(self, xadv, yadv)
        return adv_loss, adv_accu, is_adv_accu
    
    def compute_model_stat(self, x:torch.Tensor, y:torch.Tensor, eps:float, **kwargs):
        self.current_eps = eps
        # compute natural loss
        nat_loss, nat_accu, is_nat_accu = self.compute_nat_loss_and_set_BN(x, y)
        if self.enable_EDAC and self.net.training:
            self.EDAC_step((x - eps).clamp(min=self.data_min), (x + eps).clamp(max=self.data_max), x, y)
            # EDAC step updates params, thus need to recompute nat loss
            nat_loss, nat_accu, is_nat_accu = self.compute_nat_loss_and_set_BN(x, y)
        # compute PGD loss
        adv_loss, adv_accu, is_adv_accu = self.get_robust_stat_from_input_noise(eps, x, y)
        loss = self.combine_loss(nat_loss, adv_loss)
        return self.format_return(loss, nat_loss, nat_accu, is_nat_accu, adv_loss, adv_accu, is_adv_accu)

class _LabelSmoothingCrossEntropy(nn.Module):
    '''
    NLL loss with label smoothing. Used in ARoW. Taken from ARoW code.

    @param
        smoothing: float; the smoothing factor
    '''
    smoothing = property(fget=lambda self: self._smoothing, fset=lambda self, value: set_value_between(self, "_smoothing", value, 0, 1, float))

    def __init__(self, smoothing:float=0.1):
        super(_LabelSmoothingCrossEntropy, self).__init__()
        assert 0 <= smoothing <= 1
        self._smoothing = float(smoothing)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class ARoWModelWrapper(PGDModelWrapper):
    '''
    Implements Anti-Robust Weighted Regularization (ARoW) training.
    Reference: https://arxiv.org/abs/2206.03353; https://github.com/dyoony/ARoW.

    @param
        reg_weight: float; the weight of the regularization term in the ARoW loss
        smoothing: float; the smoothing factor for label smoothing

    @property
        LS_loss_fn: _LabelSmoothingCrossEntropy; the loss function with label smoothing
        LS_loss_fn.smoothing: float; the smoothing factor
        reg_weight: float; the weight of the regularization term in the ARoW loss
    '''
    reg_weight = property(fget=lambda self: self._reg_weight, fset=lambda self, value: set_value_typecast(self, "_reg_weight", value, float, lambda x: x>0, "reg_weight must be a positive float."))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, reg_weight:float=7, smoothing:float=0.2, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args, **kwargs)
        self.LS_loss_fn = _LabelSmoothingCrossEntropy(smoothing=smoothing)
        self._reg_weight = float(reg_weight)

    def get_ARoW_loss(self, inputs:torch.Tensor, adv_outputs:torch.Tensor, targets:torch.Tensor, model:Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = model(inputs)
        adv_probs = F.softmax(adv_outputs, dim=1)
        nat_probs = F.softmax(outputs, dim=1)
        true_probs = torch.gather(adv_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
        sup_loss = self.LS_loss_fn(outputs, targets)
        rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (1. - true_probs)).mean()
        return sup_loss, rob_loss

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if not self.net.training:
            # compute based on PGD
            with torch.no_grad():
                adv_loss, adv_accu, is_adv_accu = super().get_robust_stat_from_bounds(lb, ub, x, y)
        else:
            # compute based on ARoW
            xadv = adv_whitebox(self.net, x, y, lb, ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="KL")
            yadv = self.net(xadv)
            adv_accu, is_adv_accu = self._Get_Accuracy(yadv, y)
            cache_adv_xy_if_need(self, xadv, yadv)
            sup_loss, reg_loss = self.get_ARoW_loss(x, yadv, y, self.net)
            adv_loss = sup_loss + self.reg_weight * reg_loss
        return adv_loss, adv_accu, is_adv_accu

class MARTModelWrapper(PGDModelWrapper):
    '''
    Implements MART training.
    Reference: https://openreview.net/forum?id=rklOg6EFwS; Code adapted from https://github.com/dyoony/ARoW.

    @param
        reg_weight: float; the weight of the regularization term in the MART loss

    @property
        reg_weight: float; the weight of the regularization term in the MART loss
    '''
    reg_weight = property(fget=lambda self: self._reg_weight, fset=lambda self, value: set_value_typecast(self, "_reg_weight", value, float, lambda x: x>0, "reg_weight must be a positive float."))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int,...], device, args, reg_weight:float=5, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args, **kwargs)
        self._reg_weight = float(reg_weight)

    def get_MART_loss(self, inputs:torch.Tensor, adv_inputs:torch.Tensor, targets:torch.Tensor, model:Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        def Boost_CE(adv_outputs, targets):
            adv_probs = F.softmax(adv_outputs, dim=1)
            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
            new_targets = torch.where(tmp1[:, -1] == targets, tmp1[:, -2], tmp1[:, -1])
            loss =  F.cross_entropy(adv_outputs, targets) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_targets)
            return loss
        outputs = model(inputs)
        adv_outputs = model(adv_inputs)
        adv_probs = F.softmax(adv_outputs, dim=1)
        nat_probs = F.softmax(outputs, dim=1)
        true_probs = torch.gather(nat_probs, 1, (targets.unsqueeze(1)).long()).squeeze()
        sup_loss = Boost_CE(adv_outputs, targets)
        rob_loss = (F.kl_div((adv_probs+1e-12).log(), nat_probs, reduction='none').sum(dim=1) * (1. - true_probs)).mean()
        return sup_loss, rob_loss

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        xadv = adv_whitebox(self.net, x, y, lb, ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        cache_adv_xy_if_need(self, xadv, yadv)
        adv_accu, is_adv_accu = self._Get_Accuracy(yadv, y)
        sup_loss, rob_loss = self.get_MART_loss(x, xadv, y, self.net)
        adv_loss = sup_loss + self.reg_weight * rob_loss
        return adv_loss, adv_accu, is_adv_accu
    
class BoxModelWrapper(BasicModelWrapper):
    '''
    Implements IBP training
    Reference: https://arxiv.org/abs/1810.12715, https://arxiv.org/abs/2103.17268; Code: https://github.com/shizhouxing/Fast-Certified-Robust-Training/

    @param
        store_box_bounds: bool; whether to store the bounds of the box domain
        relu_shrinkage: float, optional; the shrinkage factor for ReLU layers. When set to a float between [0,1], the upper bound for unstable ReLUs will be multiplied by this factor. This behavior is introduced by SABR (https://openreview.net/forum?id=7oFuxtJtUMH).

    @property
        store_box_bounds: bool; whether to store the bounds of the box domain
        relu_shrinkage: float, optional, final; the shrinkage factor for ReLU layers
    '''
    store_box_bounds = property(fget=lambda self: self._store_box_bounds, fset=lambda self, value: set_value_typecheck(self, "_store_box_bounds", value, bool))
    relu_shrinkage = property(fget=lambda self: self._relu_shrinkage, fset=None) # read-only

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int,...], device, args, store_box_bounds:bool=False, relu_shrinkage:Optional[float]=None, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args)
        self._store_box_bounds = bool(store_box_bounds)
        if relu_shrinkage is not None:
            for layer in self.net:
                if isinstance(layer, abs_layers.ReLU):
                    layer.relu_shrinkage = relu_shrinkage
            logging.info(f"Setting ReLU shrinkage to {relu_shrinkage}")
        self._relu_shrinkage:Final = relu_shrinkage

    def get_IBP_bounds(self, abs_net:absSequential, input_lb:torch.Tensor, input_ub:torch.Tensor, y:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        If y is specified, then use final layer elision trick and return upper bounds on margin (first column padded with zero) and pseudo-labels (all zero); otherwise, return the lower and upper bounds of the output.

        @param
            abs_net: absSequential; the model to be used
            input_lb: torch.Tensor; the lower bound of the batched input data
            input_ub: torch.Tensor; the upper bound of the batched input data
            y: Optional[torch.Tensor]; the batched label
        '''
        x_abs = HybridZonotope.construct_from_bounds(input_lb, input_ub, domain='box')
        if y is None:
            abs_out = abs_net(x_abs)
            out_lb, out_ub = abs_out.concretize()
            if not self.store_box_bounds:
                abs_net.reset_bounds()
            return out_lb, out_ub
        else:
            pseudo_bound, pseudo_labels = propagate_abs(abs_net, "box", x_abs, y)
            if not self.store_box_bounds:
                abs_net.reset_bounds()
            return pseudo_bound, pseudo_labels

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        pseudo_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
        loss = self.loss_fn(pseudo_bound, pseudo_labels)
        robust_accu, is_robust_accu = self._Get_Accuracy(pseudo_bound, pseudo_labels)
        return loss, robust_accu, is_robust_accu

class DeepPolyModelWrapper(BasicModelWrapper):
    '''
    Implements DeepPoly / CROWN-IBP / CROWN-IBP(loss fusion) training

    @param
        use_dp_box: bool; whether to use CROWN-IBP. When True, all intermediate bounds will be computed using IBP.
        loss_fusion: bool; whether to use loss fusion. This will speed up the training by O(num_class) times by only estimating the upper bounds of the final loss instead of the margin.
        keep_fusion_when_test: bool; whether to keep loss fusion during evaluation

    @property
        use_dp_box: bool; whether to use CROWN-IBP
        loss_fusion: bool; whether to use loss fusion
        keep_fusion_when_test: bool; whether to keep loss fusion during evaluation

    @remark
        The loss_fusion is only applied during training and when use_dp_box=True. When DeepPoly is used, i.e., use_dp_box=False, the loss_fusion is not ignored.
        When loss_fusion=True, the robust accuracy cannot be estimated and is set to 0 during training; the robust accuracy is based on CROWN-IBP during evaluation.
        Loss fusion is not always more precise since the margin is estimated with IBP bounds instead of CROWN-IBP bounds.
    '''
    use_dp_box = property(fget=lambda self: self._use_dp_box, fset=lambda self, value: set_value_typecheck(self, "_use_dp_box", value, bool))
    loss_fusion = property(fget=lambda self: self._loss_fusion, fset=lambda self, value: set_value_typecheck(self, "_loss_fusion", value, bool))
    keep_fusion_when_test = property(fget=lambda self: self._keep_fusion_when_test, fset=lambda self, value: set_value_typecheck(self, "_keep_fusion_when_test", value, bool))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, use_dp_box:bool=False, loss_fusion:bool=False, keep_fusion_when_test=False, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args)
        self._use_dp_box = bool(use_dp_box)
        self._loss_fusion = bool(loss_fusion)
        self._keep_fusion_when_test = keep_fusion_when_test
        if self.loss_fusion and self.keep_fusion_when_test:
            assert args.model_selection in ["loss", None], "Test time loss fusion is only supported when model_selection is loss or None."

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        x_abs = HybridZonotope.construct_from_bounds(lb, ub, domain='box')
        domain = "deeppoly_box" if self.use_dp_box else "deeppoly"
        if domain == "deeppoly_box" and self.loss_fusion:
            domain = "deeppoly_box_loss_fusion"
            if not self.keep_fusion_when_test and not self.net.training:
                # loss fusion is only applied during training
                domain = "deeppoly_box"
        pseudo_bound, pseudo_labels = propagate_abs(self.net, domain, x_abs, y)
        if domain == "deeppoly_box_loss_fusion":
            loss = pseudo_bound.mean()
            # we cannot estimate robust loss from loss fusion
            robust_accu = torch.zeros(1, device=self.device)
            is_robust_accu = torch.zeros(len(x), dtype=torch.bool, device=self.device)
        else:
            loss = self.loss_fn(pseudo_bound, pseudo_labels)
            robust_accu, is_robust_accu = self._Get_Accuracy(pseudo_bound, pseudo_labels)
        return loss, robust_accu, is_robust_accu

class _MultiPGDMixin:
    '''
    Implements PGD estimation for the margin. Each dimension in the margin will be estimated by a separate PGD attack.

    @remark
        self.args.estimation_batch can be used to control the batch size for the estimation as we are now having num_class-1 times more inputs than a PGD attack.
    '''
    def _get_bound_estimation_from_pts(self, net:absSequential, pts:torch.Tensor, C:torch.Tensor) -> torch.Tensor:
        '''
        Get the estimated bounds from the pivotal points.

        @param
            net: absSequential; the model to be used
            pts: torch.Tensor; the pivotal points
            C: torch.Tensor; the query matrix constructed with y. This can be constructed by construct_C(net.output_dim[-1], y) for the final margin.

        @return
            estimated_bounds: torch.Tensor; the estimated bounds

        @remark
            The pivotal points are the adversarial examples in the latent space for the margin estimation. The estimated bounds are the upper bounds of the margin pred_i - pred_{y}. Pivot points are supposed to be constructed by _get_pivotal_points. This function will keep the gradient link from the pivotal points to the model's output. MultiPGDModelWrapper show-cases how to use this mixin.
        '''
        assert C is not None, "PGD estimation is supposed to be used for margins."
        # # main idea: convert the num_class-1 adv inputs into one batch to compute the bound at the same time; involve many reshaping
        batch_C = C.unsqueeze(1).expand(-1, pts.shape[1], -1, -1).reshape(-1, *(C.shape[1:])) # may need shape adjustment
        batch_pts = pts.reshape(-1, *(pts.shape[2:]))
        out_pts = net(batch_pts, C=batch_C)
        out_pts = out_pts.reshape(*(pts.shape[:2]), *(out_pts.shape[1:]))
        out_pts = - out_pts # the out is the lower bound of yt - yi, transform it to the upper bound of yi - yt
        # postcondition: the out_pts is in shape (batch_size, n_class - 1, n_class - 1)
        ub = torch.diagonal(out_pts, dim1=1, dim2=2) # shape: (batch_size, n_class - 1)
        estimated_bounds = torch.cat([torch.zeros(size=(ub.shape[0],1), dtype=ub.dtype, device=ub.device), ub], dim=1) # shape: (batch_size, n_class)
        return estimated_bounds

    def _get_pivotal_points_one_batch(self, net:absSequential, lb:torch.Tensor, ub:torch.Tensor, num_steps:int, restarts:int, C:torch.Tensor) -> torch.Tensor:
        '''
        Get adversarial examples in the latent space for the margin estimation and batched inputs.

        @remark
            This function is not supposed to be called directly. Use _get_pivotal_points instead.
        '''
        num_pivotal = net.output_dim[-1] - 1 # only need to estimate n_class - 1 dim for the final output

        def init_pts(input_lb, input_ub):
            rand_init = input_lb.unsqueeze(1) + (input_ub-input_lb).unsqueeze(1)*torch.rand(input_lb.shape[0], num_pivotal, *input_lb.shape[1:], device=self.device)
            return rand_init
        
        def select_schedule(num_steps):
            if num_steps >= 10 and num_steps <= 50:
                lr_decay_milestones = [int(num_steps*0.7)]
            elif num_steps > 50 and num_steps <= 80:
                lr_decay_milestones = [int(num_steps*0.4), int(num_steps*0.7)]
            elif num_steps > 80:
                lr_decay_milestones = [int(num_steps*0.3), int(num_steps*0.6), int(num_steps*0.8)]
            else:
                lr_decay_milestones = []
            return lr_decay_milestones

        # TODO: move this to args factory? Maybe not; don't want to expose too much details and increase the space for hyperparameter tuning
        lr_decay_milestones = select_schedule(num_steps)
        lr_decay_factor = 0.2
        init_lr = max(0.2, 2/num_steps) # this makes sure for num_steps<5, the attack is still able to reach the boundary

        pts = init_pts(lb, ub)
        variety = (ub - lb).unsqueeze(1).detach()
        best_estimation = -np.inf*torch.ones(pts.shape[:2], device=pts.device, dtype=pts.dtype)
        best_pts = torch.zeros_like(pts)
        with torch.enable_grad():
            for re in range(restarts):
                lr = init_lr
                pts = init_pts(lb, ub)
                for it in range(num_steps+1):
                    pts.requires_grad = True
                    estimated_pseudo_bound = self._get_bound_estimation_from_pts(net, pts, C=C)
                    improve_idx = estimated_pseudo_bound[:, 1:] > best_estimation
                    best_estimation[improve_idx] = estimated_pseudo_bound[:, 1:][improve_idx].detach().float()
                    best_pts[improve_idx] = pts[improve_idx].detach()
                    if it != num_steps:
                        # wants to maximize the estimated bound
                        loss = - estimated_pseudo_bound.sum()
                        if torch.is_autocast_enabled():
                            loss = loss * 2.**12
                        grad = torch.autograd.grad(loss, pts)[0]
                        assert not torch.isnan(grad).any(), "nan found in grad during attack; If automatic mixed precision is used, try a smaller scaling factor (usually not recommended). Otherwise, it usually indicates grad overflow due to inproper output scale."

                        new_pts = pts - grad.sign() * lr * variety
                        pts = project_to_bounds(new_pts, lb.unsqueeze(1), ub.unsqueeze(1)).detach()
                        if (it+1) in lr_decay_milestones:
                            lr *= lr_decay_factor
        return best_pts

    def _get_pivotal_points(self, net:absSequential, input_lb:torch.Tensor, input_ub:torch.Tensor, num_steps:int, restarts:int, C:torch.Tensor) -> torch.Tensor:
        '''
        Get adversarial examples in the latent space for the margin estimation.

        @param
            net: absSequential; the network to be used
            input_lb: torch.Tensor; the lower bound of the batched input data
            input_ub: torch.Tensor; the upper bound of the batched input data
            num_steps: int; the number of steps for PGD
            restarts: int; the number of restarts for PGD
            C: torch.Tensor; the query matrix constructed with y. This can be constructed by construct_C(net.output_dim[-1], y) for the final margin.

        @remark
            This assumes the net is fixed in this procedure. If a BatchNorm is involved, freeze its stat before calling this function.
        '''
        assert C is not None, "Should only estimate for the final margin"
        lb, ub = input_lb.clone().detach(), input_ub.clone().detach()

        pt_list = []
        # split into batches if required
        bs = self.args.estimation_batch
        if bs is None:
            bs = len(lb)
        lb_batches = [lb[i*bs:(i+1)*bs] for i in range(math.ceil(len(lb) / bs))]
        ub_batches = [ub[i*bs:(i+1)*bs] for i in range(math.ceil(len(ub) / bs))]
        C_batches = [C[i*bs:(i+1)*bs] for i in range(math.ceil(len(C) / bs))]
        for lb_one_batch, ub_one_batch, C_one_batch in zip(lb_batches, ub_batches, C_batches):
            pt_list.append(self._get_pivotal_points_one_batch(net, lb_one_batch, ub_one_batch, num_steps, restarts, C_one_batch))
        pts = torch.cat(pt_list, dim=0)
        return pts

class MultiPGDModelWrapper(_MultiPGDMixin, PGDModelWrapper):
    '''
    Implements PGD training with PGD estimation for the margin, each dimension estimated by a separate PGD attack, i.e., K-1 separate PGD attacks for K classes.
    '''
    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, **kwargs):
        super().__init__(net, loss_fn, input_dim, device, args, **kwargs)

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        assert y is not None, "MultiPGDModelWrapper requires the label to compute the margin."
        C = construct_C(self.net.output_dim[-1], y)

        with torch.no_grad():
            # note that the batch norm stats have been frozen during natural loss computation, thus no need to freeze again
            pts = self._get_pivotal_points(self.net, lb, ub, self.num_steps, self.restarts, C)

        pseudo_bound = self._get_bound_estimation_from_pts(self.net, pts, C)
        pseudo_labels = torch.zeros(size=(pseudo_bound.size(0),), dtype=torch.int64, device=pseudo_bound.device)
        adv_accu, is_adv_accu = self._Get_Accuracy(pseudo_bound, pseudo_labels)
        adv_loss = self.loss_fn(pseudo_bound, pseudo_labels)
        return adv_loss, adv_accu, is_adv_accu

class TAPSModelWrapper(_MultiPGDMixin, BoxModelWrapper):
    '''
    Implements TAPS training
    Reference: https://arxiv.org/abs/2305.04574; Code: https://github.com/eth-sri/taps

    @param
        block_sizes: Tuple[int, int]; the sizes of the two blocks in the net, the first for IBP, the second for PGD. The sum of the two should be equal to the length of the net.
        Other parameters inherited from BoxModelWrapper

    @property
        soft_thre: float; the threshold for the soft-thresholding. By default, it is set to args.soft_thre. 0.5 is the recommended value.
        num_steps: int; the number of steps for PGD. By default, it is set to args.train_steps.
        disable_TAPS: bool; whether to disable TAPS and use IBP only. By default, False. This will be useful during eps annealing.
        TAPS_grad_scale: float; the scale factor for the TAPS gradient (IBP gradient as 1). By default, it is set to args.taps_grad_scale. 5 is a good value but tuning is recommended.
        restarts: int; the number of restarts for PGD. By default, it is set to args.restarts.
        no_ibp_multiplier: bool; whether to disable the IBP multiplier in the TAPS loss. By default, False.
        net_blocks_abs: List[abs_layers.Sequential]; the abstract network that is split into blocks.
        Other properties inherited from BoxModelWrapper.
    '''
    latent_search_restarts = property(fget=lambda self: self._latent_search_restarts, fset=lambda self, value: set_value_typecast(self, "_latent_search_restarts", value, int, lambda x: x>0, "latent_search_restarts must be a positive integer."))
    soft_thre = property(fget=lambda self: self._soft_thre, fset=lambda self, value: set_value_between(self, "_soft_thre", value, 0, 1, float))
    latent_search_steps = property(fget=lambda self: self._latent_search_steps, fset=lambda self, value: set_value_typecast(self, "_latent_search_steps", value, int, lambda x: x>0, "latent_search_steps must be a positive integer."))
    disable_TAPS = property(fget=lambda self: self._disable_TAPS, fset=lambda self, value: set_value_typecheck(self, "_disable_TAPS", value, bool))
    TAPS_grad_scale = property(fget=lambda self: self._TAPS_grad_scale, fset=lambda self, value: set_value_typecast(self, "_TAPS_grad_scale", value, float, lambda x: x>0, "TAPS_grad_scale must be a positive float."))
    no_ibp_multiplier = property(fget=lambda self: self._no_ibp_multiplier, fset=lambda self, value: set_value_typecheck(self, "_no_ibp_multiplier", value, bool))
    net_blocks_abs = property(fget=lambda self: self._net_blocks_abs, fset=None) # read-only

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, block_sizes:Iterable[int], store_box_bounds:bool=False, relu_shrinkage:Optional[float]=None, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, store_box_bounds=store_box_bounds, relu_shrinkage=relu_shrinkage, **kwargs)
        self._net_blocks_abs = self._split_net_to_blocks(block_sizes)
        self._soft_thre = float(args.soft_thre)
        self._disable_TAPS = False # when true, TAPS is equivalent to IBP
        self._TAPS_grad_scale = float(args.taps_grad_scale)
        self._latent_search_steps = int(args.train_steps)
        self._latent_search_restarts = int(args.restarts)
        self._no_ibp_multiplier = bool(args.no_ibp_multiplier)

    def _split_net_to_blocks(self, block_sizes:Iterable[int]):
        assert block_sizes is not None and len(block_sizes) == 2, f"TAPS assume two blocks: the first uses IBP, the second uses PGD."
        assert len(self.net) == sum(block_sizes), f"Provided block splits have {sum(block_sizes)} layers, but the net has {len(self.net)} layers."
        start = 0
        blocks = []
        for size in block_sizes:
            end = start + size
            abs_block = abs_layers.Sequential(*self.net[start:end])
            abs_block.output_dim = abs_block[-1].output_dim
            blocks.append(abs_block)
            start = end
        return blocks

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if self.disable_TAPS:
            return super().get_robust_stat_from_bounds(lb, ub, x, y)
        
        # propagate the bound block-wisely
        for block_id, block in enumerate(self.net_blocks_abs):
            if block_id + 1 < len(self.net_blocks_abs):
                lb, ub = self.get_IBP_bounds(block, lb, ub)
            else:
                # prepare PGD bounds, Box bounds for y_i - y_t
                TAPS_bound = self.get_TAPS_bounds(block, lb, ub, self.latent_search_steps, self.latent_search_restarts, y)
                Box_bound, pseudo_labels = self.get_IBP_bounds(block, lb, ub, y)

        loss = _GradExpander.apply(self.loss_fn(TAPS_bound, pseudo_labels), self.TAPS_grad_scale)
        if not self.no_ibp_multiplier:
            loss = loss * self.loss_fn(Box_bound, pseudo_labels)
        robust_accu, is_robust_accu = self._Get_Accuracy(TAPS_bound, pseudo_labels)
        return loss, robust_accu, is_robust_accu

    def get_TAPS_bounds(self, block:absSequential, input_lb:torch.Tensor, input_ub:torch.Tensor, num_steps:int, restarts:int, y:torch.Tensor):
        assert y is not None, "TAPS requires the target label to perform margin estimation."
        C = construct_C(block.output_dim[-1], y)
        with torch.no_grad():
            # note that the batch norm stats have been frozen during natural loss computation, thus no need to freeze again
            pts = self._get_pivotal_points(block, input_lb, input_ub, num_steps, restarts, C)
        # Establish gradient link between pivotal points and bound
        # via rectified linear link
        pts = torch.transpose(pts, 0, 1)
        pts = _RectifiedLinearGradientLink.apply(input_lb.unsqueeze(0), input_ub.unsqueeze(0), pts, self.args.soft_thre, 1e-5)
        pts = torch.transpose(pts, 0, 1)
        bounds = self._get_bound_estimation_from_pts(block, pts, C)
        return bounds


class _RectifiedLinearGradientLink(torch.autograd.Function):
    '''
    Belongs to TAPS.

    Estabilish Rectified linear gradient link between the input bounds and the input point.
    Note that this is not a valid gradient w.r.t. the forward function
    Take ub as an example: 
        For dims that x[dim] in [lb, ub-c*(ub-lb)], the gradient w.r.t. ub is 0. 
        For dims that x[dim] == ub, the gradient w.r.t. ub is 1.
        For dims that x[dim] in [ub-c*(ub-lb), ub], the gradient is linearly interpolated between 0 and 1.
    
    x should be modified to shape (batch_size, *bound_dims) by reshaping.
    bounds should be of shape (1, *bound_dims)
    '''
    @staticmethod
    def forward(ctx, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, c:float, tol:float):
        ctx.save_for_backward(lb, ub, x)
        ctx.c = c
        ctx.tol = tol
        return x
    
    @staticmethod
    def backward(ctx, grad_x):
        lb, ub, x = ctx.saved_tensors
        c, tol = ctx.c, ctx.tol
        slackness = c * (ub - lb)
        # handle grad w.r.t. ub
        thre = (ub - slackness)
        rectified_grad_mask = (x >= thre)
        grad_ub = (rectified_grad_mask * grad_x * (x - thre).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # handle grad w.r.t. lb
        thre = (lb + slackness)
        rectified_grad_mask = (x <= thre)
        grad_lb = (rectified_grad_mask * grad_x * (thre - x).clamp(min=0.5*tol) / slackness.clamp(min=tol)).sum(dim=0, keepdim=True)
        # we don't need grad w.r.t. x and param
        return grad_lb, grad_ub, None, None, None

class _GradExpander(torch.autograd.Function):
    '''
    Belongs to TAPS.

    Multiply the gradient by alpha
    '''
    @staticmethod
    def forward(ctx, x, alpha:float=1):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_x):
        return ctx.alpha * grad_x, None

class SmallBoxModelWrapper(BoxModelWrapper):
    '''
    Implements SABR training
    Reference: https://openreview.net/forum?id=7oFuxtJtUMH; Code: https://github.com/eth-sri/SABR

    @param
        eps_shrinkage: float; the shrinkage factor for the small box. A small box with size = eps * eps_shrinkage will be used for IBP.
        relu_shrinkage: float; the shrinkage factor for the ReLU. After each ReLU, the upper bounds of unstable neurons will be multiplied by 1-relu_shrinkage. None means no shrinkage, i.e., relu_shrinkage=0.
        cache_advx: bool; whether to cache the adversarial input.
        cache_advy: bool; whether to cache the adversarial output, which is inferred from the adversarial input automatically (thus leads to overhead).
    
    @property
        eps_shrinkage: float; the shrinkage factor for the small box.
        cache_advx: bool; whether to cache the adversarial input
        cache_advy: bool; whether to cache the adversarial output
        current_advx: torch.Tensor; the current adversarial input; None if cache_advx is False
        current_advy: torch.Tensor; the current adversarial output; None if cache_advy is False
    '''
    eps_shrinkage = property(fget=lambda self: self._eps_shrinkage, fset=lambda self, value: set_value_between(self, "_eps_shrinkage", value, 0, 1, float))
    cache_advx = property(fget=lambda self: self._cache_advx, fset=lambda self, value: set_value_typecheck(self, "_cache_advx", value, bool))
    cache_advy = property(fget=lambda self: self._cache_advy, fset=lambda self, value: set_value_typecheck(self, "_cache_advy", value, bool))
    input_search_steps = property(fget=lambda self: self._input_search_steps, fset=lambda self, value: set_value_typecast(self, "_input_search_steps", value, int, lambda x: x>0, "input_search_steps must be a positive integer."))
    input_search_restarts = property(fget=lambda self: self._input_search_restarts, fset=lambda self, value: set_value_typecast(self, "_input_search_restarts", value, int, lambda x: x>0, "input_search_restarts must be a positive integer."))
    
    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, store_box_bounds:bool=False, eps_shrinkage:float=1, relu_shrinkage:Optional[float]=None, cache_advx:bool=False, cache_advy:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, store_box_bounds=store_box_bounds, relu_shrinkage=relu_shrinkage, **kwargs)
        self._eps_shrinkage = float(eps_shrinkage)
        logging.info(f"Using small box with eps shrinkage: {self._eps_shrinkage}")
        self._cache_advx = bool(cache_advx)
        self._cache_advy = bool(cache_advy)
        self.current_advx = None
        self.current_advy = None
        self._input_search_steps = int(args.train_steps)
        self._input_search_restarts = int(args.restarts)

    def get_robust_stat_from_input_noise(self, eps:float, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:    
        with torch.no_grad():
            lb_box, ub_box = (x-eps).clamp(min=self.data_min), (x+eps).clamp(max=self.data_max)
            adex = adv_whitebox(self.net, x, y, lb_box, ub_box, self.device, num_steps=self.input_search_steps, restarts=self.input_search_restarts, lossFunc="pgd")
            eff_eps = (ub_box - lb_box) / 2 * self.eps_shrinkage
            x_new = torch.clamp(adex, lb_box+eff_eps, ub_box-eff_eps)
            lb_new, ub_new = (x_new - eff_eps), (x_new + eff_eps)
            assert (lb_new >= lb_box - 1e-5).all() and (ub_new <= ub_box + 1e-5).all(), "The new box is not within the original box."
            cache_adv_xy_if_need(self, adex, None) # let advy be inferred if needed
        # do IBP for the small box
        robust_loss, robust_accu, is_robust_accu = self.get_robust_stat_from_bounds(lb_new, ub_new, x_new, y)
        return robust_loss, robust_accu, is_robust_accu
    
class STAPSModelWrapper(TAPSModelWrapper, SmallBoxModelWrapper):
    '''
    Implements STAPS training, a direct combination of TAPS and SABR
    Reference: https://arxiv.org/abs/2305.04574; Code: https://github.com/eth-sri/taps

    @param
        Everything from TAPSModelWrapper and SmallBoxModelWrapper

    @property
        Everything from TAPSModelWrapper and SmallBoxModelWrapper
    '''
    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, block_sizes:Iterable[int], store_box_bounds:bool= False, eps_shrinkage:float=1, relu_shrinkage:Optional[float]=None, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, block_sizes=block_sizes, store_box_bounds=store_box_bounds, eps_shrinkage=eps_shrinkage, relu_shrinkage=relu_shrinkage, **kwargs)
        # # overwirte with a fixed value for SABR as the default
        # self.input_search_steps = 10
        # self.input_search_restarts = 1

class MTLIBPModelWrapper(BoxModelWrapper):
    '''
    Implements MTL-IBP training; this is also the base class for EXP-IBP and CC-IBP.
    Reference: https://openreview.net/pdf?id=mzyZ4wzKlM; Code: https://github.com/alessandrodepalma/expressive-losses/

    @param
        ibp_coef: float; the coefficient for the IBP loss.
        Other parameters inherited from BoxModelWrapper

    @property
        ibp_coef: float; the coefficient for the IBP loss
        num_steps: int; the number of steps for PGD. By default, it is set to args.train_steps.
        restarts: int; the number of restarts for PGD. By default, it is set to args.restarts.
        Other properties inherited from BoxModelWrapper

    @remark
        Model selection is not supported for MTL-IBP / EXP-IBP / CC-IBP since they do not estimate certified accuracy. The returned robust accuracy is PGD accuracy. MTL-IBP / EXP-IBP / CC-IBP only differs in how they compute the robust loss.
        MTL-IBP: ibp_coef * IBP_loss + (1 - ibp_coef) * PGD_loss
        EXP-IBP: IBP_loss^ibp_coef * PGD_loss^(1-ibp_coef)
        CC-IBP: loss_fn(IBP_bound * ibp_coef + PGD_bound * (1 - ibp_coef), y)
    '''
    ibp_coef = property(fget=lambda self: self._ibp_coef, fset=lambda self, value: set_value_between(self, "_ibp_coef", value, 0, 1, float))
    num_steps = property(fget=lambda self: self._num_steps, fset=lambda self, value: set_value_typecast(self, "_num_steps", value, int, lambda x: x>0, "num_steps must be a positive integer."))
    restarts = property(fget=lambda self: self._restarts, fset=lambda self, value: set_value_typecast(self, "_restarts", value, int, lambda x: x>0, "restarts must be a positive integer."))
    step_size = property(fget=lambda self: self._step_size, fset=lambda self, value: set_value_typecast(self, "_step_size", value, float, lambda x: x>0, "step_size must be a positive float."))
    attack_range_scale = property(fget=lambda self: self._attack_range_scale, fset=lambda self, value: set_value_typecast(self, "_attack_range_scale", value, float, lambda x: x>0, "attack_range_scale must be a positive float."))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, ibp_coef:float, attack_range_scale:float, store_box_bounds:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, store_box_bounds=store_box_bounds, **kwargs)
        if not (getattr(args, "use_ccdist_training", False) and abs(float(ibp_coef)) < 1e-12):
            assert args.model_selection is None, "MTL-IBP / EXP-IBP / CC-IBP / CC-DIST does not support model selection."
        self._ibp_coef = float(ibp_coef)
        self._num_steps = int(args.train_steps)
        self._restarts = int(args.restarts)
        self._step_size = max(0.25, 2/self._num_steps) if args.step_size is None else float(args.step_size)
        self._attack_range_scale = float(attack_range_scale)
        logging.info(f"Using attack range scale: {self._attack_range_scale} and ibp coef: {self._ibp_coef}.")

    def get_attack_range(self, eps:float, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attack_eps = eps * self.attack_range_scale
        return (x - attack_eps).clamp(min=self.data_min), (x + attack_eps).clamp(max=self.data_max)

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        pseudo_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
        IBP_loss = self.loss_fn(pseudo_bound, pseudo_labels)
        attack_lb, attack_ub = self.get_attack_range(self.current_eps, x)
        xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        pgd_loss = self.loss_fn(yadv, y)
        pgd_accu, is_pgd_accu = self._Get_Accuracy(yadv, y)
        robust_loss = IBP_loss*self.ibp_coef  + pgd_loss*(1 - self.ibp_coef)
        return robust_loss, pgd_accu, is_pgd_accu

class EXPIBPModelWrapper(MTLIBPModelWrapper):
    '''
    Implements EXP-IBP training
    Reference: https://openreview.net/pdf?id=mzyZ4wzKlM; Code: https://github.com/alessandrodepalma/expressive-losses/

    @param
        Everything from MTLIBPModelWrapper

    @property
        Everything from MTLIBPModelWrapper
    '''
    def __init__(self, net, loss_fn, input_dim, device, args, ibp_coef:float, attack_range_scale:float, store_box_bounds:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, ibp_coef=ibp_coef, attack_range_scale=attack_range_scale, store_box_bounds=store_box_bounds, **kwargs)
    
    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        pseudo_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
        IBP_loss = self.loss_fn(pseudo_bound, pseudo_labels)
        attack_lb, attack_ub = self.get_attack_range(self.current_eps, x)
        xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        pgd_loss = self.loss_fn(yadv, y)
        pgd_accu, is_pgd_accu = self._Get_Accuracy(yadv, y)
        robust_loss = IBP_loss**self.ibp_coef  * pgd_loss**(1 - self.ibp_coef)
        return robust_loss, pgd_accu, is_pgd_accu

class CCIBPModelWrapper(MTLIBPModelWrapper):
    '''
    Implements CC-IBP training
    Reference: https://openreview.net/pdf?id=mzyZ4wzKlM; Code: https://github.com/alessandrodepalma/expressive-losses/

    @param
        Everything from MTLIBPModelWrapper

    @property
        Everything from MTLIBPModelWrapper
    '''
    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, ibp_coef:float, attack_range_scale:float, store_box_bounds:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, ibp_coef=ibp_coef, attack_range_scale=attack_range_scale, store_box_bounds=store_box_bounds, **kwargs)
    
    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        IBP_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
        attack_lb, attack_ub = self.get_attack_range(self.current_eps, x)
        xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        yadv = self.net(xadv)
        PGD_bound = self.convert_pred_to_margin(y, yadv)
        margin = IBP_bound*self.ibp_coef + PGD_bound*(1 - self.ibp_coef)
        robust_loss = self.loss_fn(margin, pseudo_labels)
        pgd_accu, is_pgd_accu = self._Get_Accuracy(yadv, y)
        return robust_loss, pgd_accu, is_pgd_accu

class CCDistModelWrapper(CCIBPModelWrapper):
    '''
    Implements CC-Dist training
    Reference: https://arxiv.org/pdf/2602.02626;

    @remark
        Model selection is not supported for CC-DIST since it do not estimate certified accuracy. The returned robust accuracy is PGD accuracy.
        CC-DIST: CC_IBP(ibp_coef) + 5/feat_dim * Robust_Dist(ibp_coef)
    '''
    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, ibp_coef:float, attack_range_scale:float, store_box_bounds:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, ibp_coef=ibp_coef, attack_range_scale=attack_range_scale, store_box_bounds=store_box_bounds, **kwargs)
        self._register_teacher(net, args.load_model, args.teacher_model)
        self.feature_net, self.head_net = self._split_feature_head(net)
        self.teacher_feature_net, self.teacher_head_net = self._split_feature_head(self.teacher)
        self.feat_dim = int(np.prod(self.feature_net.output_dim))

    def _register_teacher(self, model:absSequential, student_path:str, teacher_path:str):
        '''Stores a frozen copy of the teacher model'''
        self.teacher:absSequential = copy.deepcopy(model)
        if teacher_path is None:
            raise NotImplementedError("CC-DIST requires a teacher.")
        if student_path != teacher_path:
            # student has been initialised with random weights, not the teacher weights
            self.teacher.load_state_dict(torch.load(teacher_path))
            print("Loaded Teacher from: ", teacher_path)
        self.teacher.eval()

    def _split_feature_head(self, net: absSequential):
        feature_net = abs_layers.Sequential(*net[:-1])
        feature_net.output_dim = feature_net[-1].output_dim
        head_net = abs_layers.Sequential(*net[-1:])
        head_net.output_dim = head_net[-1].output_dim
        return feature_net, head_net

    def _flatten_feat(self, feat: torch.Tensor) -> torch.Tensor:
        '''Flatten to shape [batch_size, feat_dim] to make impl robust to conv layers'''
        return feat.flatten(start_dim=1)

    def get_robust_dist_loss(self, x: torch.Tensor, feat_adv: torch.Tensor, feat_lb: torch.Tensor, feat_ub: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_teacher = self._flatten_feat(self.teacher_feature_net(x))

        feat_adv = self._flatten_feat(feat_adv)
        feat_lb = self._flatten_feat(feat_lb)
        feat_ub = self._flatten_feat(feat_ub)

        feat_cc_lb = (1.0 - self.ibp_coef) * feat_adv + self.ibp_coef * feat_lb
        feat_cc_ub = (1.0 - self.ibp_coef) * feat_adv + self.ibp_coef * feat_ub

        robust_dist_loss = torch.maximum(
            (feat_cc_lb - feat_teacher) ** 2,
            (feat_cc_ub - feat_teacher) ** 2
        )

        return robust_dist_loss.sum(dim=1).mean()

    def get_robust_dist_loss_no_ibp(self, x, feat_adv):
        with torch.no_grad():
            feat_teacher = self._flatten_feat(self.teacher_feature_net(x))
        feat_adv = self._flatten_feat(feat_adv)
        return ((feat_adv - feat_teacher) ** 2).sum(dim=1).mean()

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        attack_lb, attack_ub = self.get_attack_range(self.current_eps, x)
        xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        feat_adv = self.feature_net(xadv)
        yadv = self.head_net(feat_adv)

        PGD_bound = self.convert_pred_to_margin(y, yadv)
        if self.ibp_coef > 0:
            feat_lb, feat_ub = self.get_IBP_bounds(self.feature_net, lb, ub, y=None)
            IBP_bound, pseudo_labels = self.get_IBP_bounds(self.head_net, feat_lb, feat_ub, y)
            margin = IBP_bound*self.ibp_coef + PGD_bound*(1 - self.ibp_coef)
            robust_dist_loss = self.get_robust_dist_loss(x, feat_adv, feat_lb, feat_ub)
        else:
            margin = PGD_bound
            pseudo_labels = torch.zeros(size=(PGD_bound.size(0),), dtype=torch.int64, device=PGD_bound.device)
            robust_dist_loss = self.get_robust_dist_loss_no_ibp(x, feat_adv)

        ccibp_loss = self.loss_fn(margin, pseudo_labels)
        total_loss = ccibp_loss  + 5.0/self.feat_dim * robust_dist_loss
        adv_accu, is_adv_accu = self._Get_Accuracy(yadv, y)
        return total_loss, adv_accu, is_adv_accu

class ADCERTModelWrapper(BoxModelWrapper):
    '''
    Implements AD-CERT training
    Reference: https://arxiv.org/pdf/2606.31653; Code: https://github.com/matteo-melis/ad-cert

    @param
        ibp_coef: float; the coefficient for the IBP loss.
        Other parameters inherited from BoxModelWrapper

    @property
        ibp_coef: float; the coefficient for the IBP loss
        num_steps: int; the number of steps for PGD. By default, it is set to args.train_steps.
        restarts: int; the number of restarts for PGD. By default, it is set to args.restarts.
        softlabel_attack: bool; use a softlabel PGD attack to generate x_adv.
        Other properties inherited from BoxModelWrapper

    @remark
        Model selection is not supported for AD-CERT since it do not estimate certified accuracy. The returned robust accuracy is PGD accuracy.
        AD-CERT: ibp_coef * IBP_loss + (1 - ibp_coef) * KL_DIV(natural_teacher_probs || adv_student_probs)
    '''
    ibp_coef = property(fget=lambda self: self._ibp_coef, fset=lambda self, value: set_value_between(self, "_ibp_coef", value, 0, 1, float))
    num_steps = property(fget=lambda self: self._num_steps, fset=lambda self, value: set_value_typecast(self, "_num_steps", value, int, lambda x: x>0, "num_steps must be a positive integer."))
    softlabel_attack = property(fget=lambda self: self._softlabel_attack, fset=lambda self, value: set_value_typecheck(self, "_softlabel_attack", value, bool))
    rslad = property(fget=lambda self: self._rslad, fset=lambda self, value: set_value_typecheck(self, "_rslad", value, bool))
    nat_kl = property(fget=lambda self: self._nat_kl, fset=lambda self, value: set_value_typecheck(self, "_nat_kl", value, bool))
    restarts = property(fget=lambda self: self._restarts, fset=lambda self, value: set_value_typecast(self, "_restarts", value, int, lambda x: x>0, "restarts must be a positive integer."))
    step_size = property(fget=lambda self: self._step_size, fset=lambda self, value: set_value_typecast(self, "_step_size", value, float, lambda x: x>0, "step_size must be a positive float."))
    attack_range_scale = property(fget=lambda self: self._attack_range_scale, fset=lambda self, value: set_value_typecast(self, "_attack_range_scale", value, float, lambda x: x>0, "attack_range_scale must be a positive float."))

    def __init__(self, net:absSequential, loss_fn:Callable, input_dim:Tuple[int, ...], device, args, ibp_coef:float, attack_range_scale:float, store_box_bounds:bool=False, **kwargs):
        super().__init__(net=net, loss_fn=loss_fn, input_dim=input_dim, device=device, args=args, store_box_bounds=store_box_bounds, **kwargs)
        if ibp_coef > 0:
            assert args.model_selection is None, (
                "AD-CERT/RSLAD-CERT with ibp_coef > 0 does not support meaningful model selection, "
                "because returned robust_accu is PGD accuracy, not certified accuracy."
            )
        self._ibp_coef = float(ibp_coef)
        self._num_steps = int(args.train_steps)
        self._restarts = int(args.restarts)
        self._softlabel_attack = bool(args.use_softlabel_attack)
        self._rslad = bool(args.use_rslad)
        self._nat_kl = bool(args.use_nat_kl)
        self._step_size = max(0.25, 2/self._num_steps) if args.step_size is None else float(args.step_size)
        self._attack_range_scale = float(attack_range_scale)
        self._register_teacher(net, args.load_model, args.teacher_model)
        logging.info(f"Using attack range scale: {self._attack_range_scale}, ibp coef: {self._ibp_coef}.")

    def _register_teacher(self, model:absSequential, student_path:str, teacher_path:str):
        '''Stores a frozen copy of the teacher model'''
        self.teacher:absSequential = copy.deepcopy(model) #TODO: Make more robust to handle different architecture to the student
        if teacher_path is None:
            raise NotImplementedError("ADCERT requires a teacher.")
        if student_path != teacher_path:
            # student has been initialised with random weights, not the teacher weights
            self.teacher.load_state_dict(torch.load(teacher_path))
            print("Loaded Teacher from: ", teacher_path)
        self.teacher.eval()

    def get_attack_range(self, eps:float, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attack_eps = eps * self.attack_range_scale
        return (x - attack_eps).clamp(min=self.data_min), (x + attack_eps).clamp(max=self.data_max)

    def get_robust_stat_from_bounds(self, lb:torch.Tensor, ub:torch.Tensor, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        assert self.rslad + self.nat_kl <= 1, "Cannot use two different distillation schemes."
        if self.ibp_coef > 0:
            pseudo_bound, pseudo_labels = self.get_IBP_bounds(self.net, lb, ub, y)
            IBP_loss = self.loss_fn(pseudo_bound, pseudo_labels)
        attack_lb, attack_ub = self.get_attack_range(self.current_eps, x)
        # Use KL attack only during training for RSLAD / soft-label attack, then
        # use CE-PGD during evaluation/model selection for comparable robust_accu.
        use_kl_attack = self.net.training and (self.softlabel_attack or self.rslad)
        if use_kl_attack:
            xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="KL", teacher=self.teacher)
        else:
            xadv = adv_whitebox(self.net, x, y, attack_lb, attack_ub, self.device, self.num_steps, step_size=self.step_size, restarts=self.restarts, lossFunc="pgd")
        s_adv_out = self.net(xadv)
        teacher_x = xadv.detach() if self.args.teacher_input == "adv" else x
        dist_loss = self.get_dist_loss(x, s_adv_out, teacher_x)
        adv_accu, is_adv_accu = self._Get_Accuracy(s_adv_out, y)
        if self.ibp_coef > 0:
            total_loss = IBP_loss*self.ibp_coef  + dist_loss*(1 - self.ibp_coef)
        else:
            total_loss = dist_loss
        return total_loss, adv_accu, is_adv_accu

    def get_dist_loss(self, inputs:torch.Tensor, s_adv_outputs:torch.Tensor, teacher_inputs: Optional[torch.Tensor]=None) -> torch.Tensor:
        if teacher_inputs is None:
            teacher_inputs = inputs
        with torch.no_grad():
            t_outputs = self.teacher(teacher_inputs)
            t_nat_probs = F.log_softmax(t_outputs, dim=1)
        if self.nat_kl:
            s_outputs = self.net(inputs)
            s_nat_probs = F.log_softmax(s_outputs, dim=1)
            return F.kl_div(s_nat_probs, t_nat_probs, log_target=True, reduction='batchmean')
        elif not self.rslad:
            s_adv_probs = F.log_softmax(s_adv_outputs, dim=1)
            return F.kl_div(s_adv_probs, t_nat_probs, log_target=True, reduction='batchmean')
        else:
            s_adv_probs = F.log_softmax(s_adv_outputs, dim=1)
            s_outputs = self.net(inputs)
            s_nat_probs = F.log_softmax(s_outputs, dim=1)
            adv_dist_coef = 5.0/6.0 # recommended dist coeff value in RSLAD paper
            return (1-adv_dist_coef) * F.kl_div(s_nat_probs, t_nat_probs, log_target=True, reduction='batchmean') + adv_dist_coef * F.kl_div(s_adv_probs, t_nat_probs, log_target=True, reduction='batchmean')

# Function wrappers
class BasicFunctionWrapper(BasicModelWrapper):     
    '''
    The base class for function wrappers. A function wrapper wraps a model wrapper and provides additional functionalities, e.g., gradient accumulation with batch norm supported by the full batch.

    @param
        model_wrapper: BasicModelWrapper; the model wrapper to be wrapped

    @property
        wrapper: BasicModelWrapper; the model wrapper inside the function wrapper
        attribute_list: List[str]; the list of attributes that should be broadcasted to the model wrapper. By default, it contains `max_eps`, `robust_weight`, `summary_accu_stat`, `freeze_BN`, `store_box_bounds`. New attributes should be added to the list if they are to be broadcasted. Broadcast is done by calling broadcast_attributes() and will only broadcast the attributes that are present in both the function wrapper and the model wrapper.

    @remark
        The function wrapper should be used as a model wrapper in the training loop. The user should set the attributes of the function wrapper, and the function wrapper will broadcast those listed in `attribute_list` to the model wrapper.
    '''
    attribute_list = property(fget=lambda self: self._attribute_list, fset=None) # read-only
    
    def __init__(self, model_wrapper:BasicModelWrapper):
        super().__init__(model_wrapper.net, model_wrapper.loss_fn, model_wrapper.input_dim, model_wrapper.device, model_wrapper.args, (model_wrapper.data_min, model_wrapper.data_max))
        self.wrapper:BasicModelWrapper = model_wrapper
        self._attribute_list = ["max_eps", "robust_weight", "summary_accu_stat", "freeze_BN", "store_box_bounds"]
        for attr in self._attribute_list:
            if hasattr(model_wrapper, attr):
                try:
                    setattr(self, attr, getattr(model_wrapper, attr)) # try to syncronoze default values to the function wrapper to avoid undesired attribute broadcast
                except:
                    pass # ignore invalid syncronization to function wrapper since this means invalid attribute value in the model wrapper; calling the model wrapper will raise the error if the attribute is still invalid at that time

    def broadcast_attributes(self):
        '''
        Syncronize the attributes of the model wrapper with the function wrapper if both have the corresponding field. Syncronized attributes are listed in self.attribute_list.

        @remark
            This is required as the user will treat a functional wrapper as a model wrapper in the training loop. Thus, functional wrappers will receive the updated attributes from users, and should broadcast them to the model wrapper.
        '''
        for attr in self._attribute_list:
            if hasattr(self.wrapper, attr):
                setattr(self.wrapper, attr, getattr(self, attr))

class GradAccuFunctionWrapper(BasicFunctionWrapper):
    '''
    Implements gradient accumulation for a BasicModelWrapper instance.
    It divides the batch into specified size, compute gradient of each subbatch, and then merge the gradients together.
    If a BN model is provided, BN stat is set based on the whole batch instead of the divided batches to ensure consistency of results.

    @param
        model_wrapper: BasicModelWrapper; the model wrapper to be wrapped

    @property
        grad_accu_batch: int; the size of the accumulation batch. By default, it is set to args.grad_accu_batch.
        disable_accumulation: bool; whether to disable accumulation. By default, False. When True, the model wrapper will behave as if it is not wrapped. Use this to check validity of the result only.
        random_state: int; the random state for the gradient accumulation. By default, None. If not None, the random state will be set before each accumulation batch. Not supposed to be set other than testing.

    @remark
        After calling compute_model_stat, the gradients are already computed and stored as param.grad. A dummy differentiable loss is returned to ensure the backward() call is valid, but this dummy backward call will do nothing. The user should call optimizer.zero_grad() before calling compute_model_stat and don't do this again after calling compute_model_stat.
    '''
    grad_accu_batch = property(fget=lambda self: self._grad_accu_batch, fset=lambda self, value: set_value_typecast(self, "_grad_accu_batch", value, int, lambda x: x>0, "grad_accu_batch must be a positive integer."))
    disable_accumulation = property(fget=lambda self: self._disable_accumulation, fset=lambda self, value: set_value_typecheck(self, "_disable_accumulation", value, bool))

    def __init__(self, model_wrapper:BasicModelWrapper, args):
        super().__init__(model_wrapper)
        self._grad_accu_batch = int(args.grad_accu_batch)
        self.named_grads = {} # used to keep the grad of each accumulation batch
        self._disable_accumulation = False
        for key, p in self.net.named_parameters():
            self.named_grads[key] = 0.0
        logging.info(f"Using gradient accumulation with batch size: {self._grad_accu_batch}")

    def compute_model_stat(self, x:torch.Tensor, y:torch.Tensor, eps:float, **kwargs):
        self.broadcast_attributes()
        self.current_eps = eps
        if self.disable_accumulation or len(x)<=self.grad_accu_batch:
            result = self.wrapper.compute_model_stat(x, y, eps, **kwargs)
            return result
        # set BN stat based on the whole batch
        self.wrapper.freeze_BN = False
        nat_loss, nat_accu, is_nat_accu = self.wrapper.compute_nat_loss_and_set_BN(x, y, **kwargs)
        self.wrapper.freeze_BN = True
        # split into batches
        num_accu_batches = math.ceil(len(x) / self.grad_accu_batch)
        is_robust_accu = []
        robust_loss = []
        retain_graph = True if len(self.wrapper.BNs) > 0 else False
        summary_accu_stat = self.wrapper.summary_accu_stat # store the original value
        self.wrapper.summary_accu_stat = False
        for i in range(num_accu_batches):
            batch_x = x[i*self.grad_accu_batch:(i+1)*self.grad_accu_batch]
            batch_y = y[i*self.grad_accu_batch:(i+1)*self.grad_accu_batch]
            # this results in a repeated and unnecessary computation of natural loss, but is affordable until a better solution
            (loss, _, batch_robust_loss), _, (_, batch_is_robust_accu) = self.wrapper.compute_model_stat(batch_x, batch_y, eps, **kwargs)
            is_robust_accu.append(batch_is_robust_accu)
            robust_loss.append(batch_robust_loss.item())
            if self.net.training:
                self.grad_cleaner.zero_grad()
                # handle autocast
                if torch.is_autocast_enabled():
                    loss = self.grad_scaler.scale(loss)
                with torch.autocast(device_type=self.device, enabled=False):
                    loss.backward(retain_graph=retain_graph)
                for key, p in self.net.named_parameters():
                    if p.grad is not None:
                        self.named_grads[key] += p.grad * len(batch_x) / len(x) # WARNING: this assumes the loss is averaged over the batch
                        p.grad = None # Clean during the procedure to avoid memory overhead. TODO: this should have no effect on max memory cost
        self.wrapper.summary_accu_stat = summary_accu_stat # restore the original value
        # merge the gradients and statistics
        is_robust_accu = torch.cat(is_robust_accu)
        if self.net.training:
            for key, p in self.net.named_parameters():
                if not isinstance(self.named_grads[key], float):
                    p.grad = self.named_grads[key]
                    self.named_grads[key] = 0.0
        robust_loss = torch.mean(torch.tensor(robust_loss)).to(x.device) # no grad here
        robust_accu = (is_robust_accu.sum() / len(is_robust_accu)).to(x.device)
        loss = self.wrapper.combine_loss(nat_loss, robust_loss).detach()
        loss.requires_grad=True # ensure loss.backward() is valid but will do nothing
        return self.format_return(loss, nat_loss, nat_accu, is_nat_accu, robust_loss, robust_accu, is_robust_accu)

class WeightSmoothFunctionWrapper(BasicFunctionWrapper):
    '''
    Implements weight smoothing. Reference: https://arxiv.org/abs/2311.00521

    First add a random small perturbation to the model weight according to the scaled standard deviation of the weight, then do a normal step w.r.t. the perturbed model. After the step, remove the perturbation.
    Formally, w_{t+1} = w_t - eta * grad(L(w_t + noise)), where noise ~ N(0, std_scale * std(w_t)).

    @param
        std_scale: float; the scale factor for the perturbation
        reset_noise: bool; whether to reset the noise after the step
    
    @property
        std_scale: float; the scale factor for the perturbation
        reset_noise: bool; whether to reset the noise after the step

    @remark
        Expected to converges to a flater minima and thus improve generalization.
    '''
    std_scale = property(fget=lambda self: self._std_scale, fset=lambda self, value: set_value_typecast(self, "_std_scale", value, float, lambda x: x>0, "std_scale must be a positive float."))

    def __init__(self, model_wrapper:BasicModelWrapper, std_scale:float):
        super().__init__(model_wrapper)
        self._std_scale = float(std_scale)
        self._rng_state = None
        self._rng_generator = torch.Generator(device=self.device)
        self._current_stds:List[torch.Tensor] = []
        self._layer_with_weights = [layer for layer in self.net if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)]

    def _perturb_weights(self):
        '''
        Add a Gaussian noise to the weights of Conv2d and Linear layers. The noise is scaled by the standard deviation of the weights and self.std_scale. This will modify the weights in-place. A generator state is kept to allow generating the same noise for the same layer later to reduce memory overhead.
        '''
        with torch.no_grad():
            self._current_stds = [torch.std(layer.weight) for layer in self._layer_with_weights]
            self._rng_state = self._rng_generator.get_state()
            for layer,std in zip(self._layer_with_weights, self._current_stds):
                noise = self.std_scale * std * torch.empty_like(layer.weight).normal_(generator=self._rng_generator)
                layer.weight.data += noise

    def compute_model_stat(self, x:torch.Tensor, y:torch.Tensor, eps:float, **kwargs):
        '''
        Perturb the weights, then call the wrapped model to compute the model stat.
        '''
        self.broadcast_attributes()
        self.current_eps = eps
        if self.net.training:
            self._perturb_weights()
        return self.wrapper.compute_model_stat(x, y, eps, **kwargs)
    
    def param_postprocess(self):
        '''
        Remove the noise from the weights before the original parameter postprocessing.
        '''
        assert self._rng_state is not None, "The noise state is not set yet. Call compute_model_stat first."
        self._rng_generator.set_state(self._rng_state)
        for layer,std in zip(self._layer_with_weights, self._current_stds):
            noise = self.std_scale * self.current_lr * std * torch.empty_like(layer.weight).normal_(generator=self._rng_generator)
            layer.weight.data -= noise
        self.wrapper.param_postprocess()

class SAMFunctionWrapper(BasicFunctionWrapper):
    '''
    Implements Sharpness-Aware Minimization (SAM) training

    Reference: https://arxiv.org/abs/2010.01412;

    @param
        rho: float; the scale factor for the sharpness penalty
        adaptive_rho: bool; whether to adaptively scale the rho based on the current epsilon; if True, the rho will be scaled by (1 - 0.99 * current_eps / max_eps)

    @property
        rho: float; the scale factor for the sharpness penalty
        adaptive_rho: bool; whether to adaptively scale the rho based on the current epsilon

    @remark
        Expected to converges to a flater minima and thus improve generalization.
    '''
    rho = property(fget=lambda self: self._rho, fset=lambda self, value: set_value_typecast(self, "_rho", value, float, lambda x: x>0, "rho must be a positive float."))
    adaptive_rho = property(fget=lambda self: self._adaptive_rho, fset=lambda self, value: set_value_typecheck(self, "_adaptive_rho", value, bool))

    def __init__(self, model_wrapper:BasicModelWrapper, rho:float, adaptive_rho:bool=False):
        super().__init__(model_wrapper)
        self._rho = float(rho)
        self._adaptive_rho = bool(adaptive_rho)
        self._pert_dict = {}

    def compute_model_stat(self, x:torch.Tensor, y:torch.Tensor, eps:float, **kwargs):
        self.broadcast_attributes()
        self.current_eps = eps
        if self.net.training:
            self.net.eval()
            # compute the adversarial parameter perturbation direction
            loss = self.wrapper.compute_model_stat(x, y, eps, **kwargs)[0][0]
            loss.backward()
            self.net.train()
            # take a small step towards the adversarial direction
            for k, p in self.net.named_parameters():
                if p.grad is None:
                    continue
                rho = self.rho if not self._adaptive_rho else self.rho * (1 - 0.99 * self.current_eps / self.max_eps)
                pert = p.grad / (torch.norm(p.grad) + 1e-12) * rho
                self._pert_dict[k] = pert
                p.data += pert
                p.grad = None
        result =  self.wrapper.compute_model_stat(x, y, eps, **kwargs)
        return result
    
    def param_postprocess(self):
        # remove the perturbation applied in compute_model_stat and reset the perturbation cache
        for k, p in self.net.named_parameters():
            if k in self._pert_dict.keys():
                p.data -= self._pert_dict[k]
                self._pert_dict[k] = 0.0
        self.wrapper.param_postprocess()


if __name__ == "__main__":
    '''
    Test init functions
    '''
    from networks import get_network
    from loaders import get_loaders
    import argparse
    from AIDomains.abstract_layers import Sequential
    logging.basicConfig(level=logging.INFO)

    device = "cpu"
    net = get_network("cnn_3layer_bn", "mnist", device)
    loss_fn = nn.CrossEntropyLoss()
    input_dim = (1, 28, 28)
    net = Sequential.from_concrete_network(net, input_dim)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    eps = 0.1
    args.train_eps = args.test_eps = 0.1

    bs = 16
    x = torch.rand(bs, *input_dim).to(device)
    y = torch.randint(10, (bs, )).flatten().to(device)
    print(x.shape, y.shape)

    args.pgd_weight = 0.5
    args.train_steps = 10
    model_wrapper = PGDModelWrapper(net, loss_fn, input_dim, device, args)
    print(model_wrapper.net(x, C=construct_C(10, y))[0])
    model_wrapper.convert_pred_to_margin(y, model_wrapper.net(x))
    

    # model_wrapper = BoxModelWrapper(net, loss_fn, input_dim, device, args, True)

    # args.soft_thre = 0.5
    # args.train_steps = 10
    # args.robust_weight = 0.5
    # args.estimation_batch = 16
    # args.TAPS_grad_scale = 0.5
    # model_wrapper = TAPSModelWrapper(net, loss_fn, input_dim, device, args, True, [6, 3])

    # args.relu_shrinkage = 0.8
    # args.robust_weight = 0.5
    # model_wrapper = SmallBoxModelWrapper(net, loss_fn, input_dim, device, args, True, 0.4)

    # args.relu_shrinkage = 0.8
    # args.robust_weight = 0.5
    # args.soft_thre = 0.5
    # args.train_steps = 10
    # args.estimation_batch = 16
    # args.TAPS_grad_scale = 0.5
    # model_wrapper = STAPSModelWrapper(net, loss_fn, input_dim, device, args, True, [6,3], 0.4)
    # model_wrapper.disable_TAPS = True

    # args.grad_accu_batch = 4
    # gc_model_wrapper = GradAccuFunctionWrapper(model_wrapper, args)

    # (loss, nat_loss, robust_loss), (nat_accu, robust_accu) = model_wrapper.compute_model_stat(x, y, eps)
