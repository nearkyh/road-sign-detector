import torch


class OptimizerNameError(Exception):

    def __str__(self):
        return "optimizer name should be one of: "\
            "Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, Rprop, RMSprop, SGD"


def from_optimizer(name, params, lr, **kwargs):
    name = name.lower()

    for k, v in kwargs.items():
        globals()[k] = v

    if name == 'Adadelta'.lower():
        try: _rho = rho
        except NameError: _rho = 0.9
        try: _eps = eps
        except NameError: _eps = 1e-6
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        optimizer = torch.optim.Adadelta(params, lr, _rho, _eps, _weight_decay)

    elif name == 'Adagrad'.lower():
        try: _lr_decay = lr_decay
        except NameError: _lr_decay = 0
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        try: _initial_accumulator_value = initial_accumulator_value
        except NameError: _initial_accumulator_value = 0
        try: _eps = eps
        except NameError: _eps = 1e-10
        optimizer = torch.optim.Adagrad(params, lr, _lr_decay, _weight_decay, _initial_accumulator_value, _eps)

    elif name == 'Adam'.lower():
        try: _betas = betas
        except NameError: _betas = (0.9, 0.999)
        try: _eps = eps
        except NameError: _eps = 1e-8
        optimizer = torch.optim.Adam(params, lr, _betas, _eps)

    elif name == 'AdamW'.lower():
        try: _betas = betas
        except NameError: _betas = (0.9, 0.999)
        try: _eps = eps
        except NameError: _eps = 1e-8
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 1e-2
        try: _amsgrad = amsgrad
        except NameError: _amsgrad = False
        optimizer = torch.optim.AdamW(params, lr, _betas, _eps, _weight_decay, _amsgrad)

    elif name == 'SparseAdam'.lower():
        try: _betas = betas
        except NameError: _betas = (0.9, 0.999)
        try: _eps = eps
        except NameError: _eps = 1e-8
        optimizer = torch.optim.SparseAdam(params, lr, _betas, _eps)

    elif name == 'Adamax'.lower():
        try: _betas = betas
        except NameError: _betas = (0.9, 0.999)
        try: _eps = eps
        except NameError: _eps = 1e-8
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        optimizer = torch.optim.Adamax(params, lr, _betas, _eps, _weight_decay)

    elif name == 'ASGD'.lower():
        try: _lambd = lambd
        except NameError: _lambd = 1e-4
        try: _alpha = alpha
        except NameError: _alpha = 0.75
        try: _t0 = t0
        except NameError: _t0 = 1e6
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        optimizer = torch.optim.ASGD(params, lr, _lambd, _alpha, _t0, _weight_decay)

    elif name == 'LBFGS'.lower():
        optimizer = torch.optim.LBFGS(params, lr)

    elif name == 'Rprop'.lower():
        optimizer = torch.optim.Rprop(params, lr)

    elif name == 'RMSprop'.lower():
        try: _alpha = alpha
        except NameError: _alpha = 0.99
        try: _eps = eps
        except NameError: _eps = 1e-8
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        try: _momentum = momentum
        except NameError: _momentum = 0
        try: _centered = centered
        except NameError: _centered = False
        optimizer = torch.optim.RMSprop(params, lr, _alpha, _eps, _weight_decay, _momentum, _centered)

    elif name == 'SGD'.lower():
        try: _momentum = momentum
        except NameError: _momentum = 0
        try: _dampening = dampening
        except NameError: _dampening = 0
        try: _weight_decay = weight_decay
        except NameError: _weight_decay = 0
        try: _nesterov = nesterov
        except NameError: _nesterov = 0
        optimizer = torch.optim.SGD(params, lr, _momentum, _dampening, _weight_decay, _nesterov)

    else:
        raise OptimizerNameError()

    return optimizer
    

if __name__ == "__main__":
    from torchvision.models import resnet34
    from torchvision.models import ResNet34_Weights

    resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    optimizer = from_optimizer(
        name='Adadelta', 
        params=resnet.parameters(), 
        lr=0.001, 
        rho=0.1, 
        weight_decay=0.1
    )
    print(optimizer.defaults)