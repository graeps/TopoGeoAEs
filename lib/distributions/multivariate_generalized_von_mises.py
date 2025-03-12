import torch
from torch.distributions import constraints, register_kl
from .parallelogram_uniform import CubeUniform
from . import VonMisesFisher


class MGVonMises(torch.distributions.Distribution):
    def __init__(self, loc, scale, precision, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.precision = precision
        self.device = loc.device
        self.d = loc.shape[-1]  # Sample dimension

        super().__init__(self.loc.size(), validate_args=validate_args)

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "precision": constraints.positive_definite
    }

    support = constraints.real
    has_rsample = True

    # TODO
    def log_prob(self, phi):
        return

    # TODO
    def log_normalizer(self, phi):
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

        # Compute a better factor (upper bound)
        max_log_prob = self._log_unnormalized_prob(self.loc)  # Peak probability at mode
        factor = torch.exp(max_log_prob).max()
        # factor = torch.exp(self.scale.sum(-1, keepdim=True))
        while not done.all():
            proposal_sample = proposal_distr.sample(shape)
            u_sample = uniform_distr.sample(shape)
            log_prob = self._log_unnormalized_prob(proposal_sample)

            accept = u_sample <= torch.exp(log_prob - torch.log(factor))

            output[accept & ~done] = proposal_sample[accept & ~done]
            done |= accept

        self.__sample = output
        return self.__sample

    def _rejection_sample2(self, shape):
        done = torch.zeros(shape, dtype=torch.bool, device=self.device)
        output = torch.zeros(shape, device=self.device)
        proposal_distr = torch.distributions.VonMises(self.loc, self.scale)
        uniform_distr = torch.distributions.Uniform(0.0, 1.0)
        l_min = torch.min(torch.linalg.eigvals(self.precision).real, dim=1).values.unsqueeze(1)
        log_scale = - l_min * self.d / 2

        while not done.all():
            proposal_sample = proposal_distr.sample(torch.Size([1])).squeeze(0) % (2 * torch.pi)
            u_sample = uniform_distr.sample(shape)
            log_prob = self._log_unnormalized_prob(proposal_sample)
            log_prob_proposal = torch.bmm(self.scale.unsqueeze(1),
                                          torch.cos(proposal_sample - self.loc).unsqueeze(2)).squeeze(-1)
            accept = torch.log(u_sample) <= log_prob - log_scale - log_prob_proposal

            output[accept & ~done] = proposal_sample[accept & ~done]
            done |= accept

    def _rejection_sample3(self, shape):
        done = torch.zeros(shape, dtype=torch.bool, device=self.device)
        output = torch.zeros(shape, device=self.device)

        x = torch.cos(self.loc)  # Shape: [batch_size, num_angles]
        y = torch.sin(self.loc)  # Shape: [batch_size, num_angles]
        loc_cartesian = torch.stack((x, y), dim=-1)  # Shape: [batch_size, num_angles, 2]
        proposal_distr = VonMisesFisher(loc_cartesian, self.scale.unsqueeze(-1))

        uniform_distr = torch.distributions.Uniform(0.0, 1.0)

        l_min = torch.min(torch.linalg.eigvals(self.precision).real, dim=1).values.unsqueeze(1)
        log_scale = - l_min * self.d / 2

        while not done.all():
            proposal_sample_cartesian = proposal_distr.rsample()
            proposal_sample = torch.atan2(proposal_sample_cartesian[:, :, 1], proposal_sample_cartesian[:, :, 0]) % (
                    2 * torch.pi)
            u_sample = torch.clamp(uniform_distr.sample(shape), min=1e-10)
            log_prob = self._log_unnormalized_prob(proposal_sample)
            log_prob_proposal = torch.bmm(self.scale.unsqueeze(1),
                                          torch.cos(proposal_sample - self.loc).unsqueeze(2)).squeeze(-1)
            accept = torch.log(u_sample) <= log_prob - log_scale - log_prob_proposal

            output[accept & ~done] = proposal_sample[accept & ~done]
            done |= accept
        return output

    def rsample(self, shape=None):
        shape = torch.Size(shape) if shape is not None else torch.Size()
        sample = self._rejection_sample(shape)
        return sample - self.scale

    # TODO
    def entropy(self):
        return


@register_kl(MGVonMises, CubeUniform)
def _kl_vmf_uniform(mgvm, cu):
    return -mgvm.entropy() + cu.entropy()
