import torch
import torch.nn as nn

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies."""
    def __init__(self, task_num=2, device='cuda:0'):
        super(AbsWeighting, self).__init__()
        self.task_num = task_num
        self.device = device
        self.rep_grad = False  # 这是用于控制是否计算表示层梯度的标志

    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods."""
        pass

    def get_share_params(self):
        # 这里假设共享参数是 sub_matrix2，它是可调节的
        shared_params = [param for param in self.parameters() if param.requires_grad]
        return shared_params

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim).to(self.device)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        else:
            # Add implementation if needed for your case
            pass
        return grads

    def zero_grad_share_params(self):
        for param in self.get_share_params():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    def _get_grads(self, losses, mode='backward'):
        if self.rep_grad:
            # This branch is for computing gradients of representations
            pass
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        if self.rep_grad:
            # This branch is for resetting gradients of representations
            pass
        else:
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    def backward(self, losses, **kwargs):
        pass
