
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.special import eval_genlaguerre as L 



class KLDLoss(nn.Module):
    def __init__(self, tilt, nz):
        super(KLDLoss, self).__init__()
        if tilt != None:
            # print('optimizing for min kld')
            self.mu_star = kld_min(tilt, nz)
            # print('mu_star: {:.3f}'.format(self.mu_star))
        else:
            self.mu_star = None

        self.nz = nz
    def forward(self, mu, logvar, ood=False):
        # kld loss options
        if self.mu_star == None:
            # kld = -1/2*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), (1))
            kld = -1/2*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            mu_norm = torch.linalg.norm(mu, dim=1)
            kld = 1/2*torch.square(mu_norm - self.mu_star)
        if not ood:
            kld = torch.mean(kld)

        return kld
    


# function definitions 
def kld(mu, tau, d):
    # no need to include z, since we run gradient descent...
    return -tau*np.sqrt(np.pi/2)*L(1/2, d/2 -1, -(mu**2)/2) + (mu**2)/2

# convex optimization problem
def kld_min(tau, d):
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3

    # inital guess (very close to optimal value)
    x = np.sqrt(max(tau**2 - d, 0))

    # run gradient descent (kld is convex)
    for step in steps:
        for i in range(1000): # TODO update this to 10000
            y1 = kld(x-dx/2, tau, d)
            y2 = kld(x+dx/2, tau, d)

            grad = (y2-y1)/dx
            x -= grad*step

    return x
