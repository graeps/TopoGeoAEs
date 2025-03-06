import torch
from torch.distributions import constraints


class MGVonMises(torch.distributions.Distribution):
    def __init__(self, loc, scale, precision, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.precision = precision
        self.device = loc.device

        super().__init__(self.loc.size(), validate_args=validate_args)

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "precision": constraints.positive_definite
    }

    support = constraints.real
    has_rsample = True

    def log_prob(self, phi):
        return

    def _log_unnormalized_prob(self, phi):
        phi = phi.unsqueeze(0) if phi.dim() == 1 else phi
        w = self.precision.unsqueeze(0) if self.precision.dim() == 2 else self.precision
        v = torch.cat((torch.cos(phi), torch.sin(phi)), dim=-1)
        v_transp = v.unsqueeze(1)  # [batch, 1, 2d]
        vw = torch.bmm(v_transp, w)  # precision shape: [batch, 2d, 2d]
        vwv = torch.bmm(vw, v.unsqueeze(2)).squeeze(-1)
        return (self.scale * torch.cos(phi - self.loc)).sum(-1, keepdim=True) - 0.5 * vwv

    def _rejection_sample(self, shape):
        done = torch.zeros(shape, dtype=torch.bool, device=self.device)
        output = torch.zeros(shape, device=self.device)
        proposal_distr = torch.distributions.Uniform(0.0, 2.0 * torch.pi)
        uniform_distr = torch.distributions.Uniform(0.0, 1.0)
        log_scale = self.scale.sum(-1, keepdim=True)

        while not done.all():
            proposal_sample = proposal_distr.sample(shape)
            u_sample = uniform_distr.sample(shape)
            log_prob = self._log_unnormalized_prob(proposal_sample)
            accept = torch.log(u_sample) <= log_prob - log_scale

            output[accept & ~done] = proposal_sample[accept & ~done]
            done |= accept
            print(done)
        return output

    def rsample(self, shape=None):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        return self._rejection_sample(shape)
