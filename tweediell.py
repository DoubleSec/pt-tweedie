import torch
from torch import log, exp, lgamma

PI = torch.tensor(3.14159265359)

# Parameter order is always x, p, phi, mu, j


def _theta(p, mu):

    return (mu ** (1 - p)) / (1 - p)


def _kappa(p, mu):

    return (mu ** (2 - p)) / (2 - p)


def _j_max(x, p, phi):

    return (x ** (2 - p)) / ((2 - p) * phi)


def _z(x, p, phi):

    alpha = (2 - p) / (1 - p)

    num = x**-alpha * (p - 1) ** alpha
    den = phi ** (1 - alpha) * (2 - p)

    return num / den


def _log_w_j(x, p, phi, j):

    z = _z(x, p, phi)
    alpha = (2 - p) / (1 - p)

    return j * log(z) - lgamma(1 + j) - lgamma(-alpha * j)


def _log_big_w(x, p, phi):

    j_max = _j_max(x, p, phi)

    sum_range = [1, 50]

    j = torch.arange(*sum_range).unsqueeze(0)

    log_w_max = _log_w_j(x, p, phi, j_max)

    log_w = _log_w_j(x, p, phi, j)

    print(log_w)

    log_w_max = log_w.max(dim=-1)[0]
    print(f"logWmax: {log_w_max}")

    s = exp(_log_w_j(x, p, phi, j) - log_w_max).sum(dim=-1)
    print(f"logw: {log(s)}")

    return log_w_max + log(s)


def tweedie_ll(x, p, phi, mu):

    theta = _theta(p, mu)
    kappa = _kappa(p, mu)

    print(f"theta: {theta}")
    print(f"kappa: {kappa}")
    print(f"alpha: {(2 - p) / (1 - p)}")
    print(f"z: {_z(x, p, phi)}")
    print(f"jmax: {_j_max(x, p, phi)}")

    log_w = _log_big_w(x, p, phi)

    log_a = -log(x) + log_w

    t2 = (phi**-1) * (x * theta - kappa)

    return log_a + t2


if __name__ == "__main__":

    # This tweedie from here: https://github.com/thequackdaddy
    from tweedie import tweedie

    x = torch.tensor(5.0).unsqueeze(0)
    p = torch.tensor(1.6).unsqueeze(0)
    phi = torch.tensor(1.0).unsqueeze(0)
    mu = torch.tensor(10.0).unsqueeze(0)

    print(tweedie.logpdf(x, p, mu, phi))

    print("--------")

    x.requires_grad = True
    p.requires_grad = True
    phi.requires_grad = True
    mu.requires_grad = True

    print(tweedie_ll(x, p, phi, mu))
