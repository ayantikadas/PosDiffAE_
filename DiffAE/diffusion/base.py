"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

from model.unet_autoenc import AutoencReturn
from config_base import BaseConfig
import enum
import math

import numpy as np
import torch as th
from model import *
from model.nn import mean_flat
from typing import NamedTuple, Tuple
from choices import *
from torch.cuda.amp import autocast
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GaussianDiffusionBeatGansConfig(BaseConfig):
    gen_type: GenerativeType
    betas: Tuple[float]
    model_type: ModelType
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    rescale_timesteps: bool
    fp16: bool
    train_pred_xstart_detach: bool = True

    def make_sampler(self):
        return GaussianDiffusionBeatGans(self)


class GaussianDiffusionBeatGans:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, conf: GaussianDiffusionBeatGansConfig,r=None,theta=None,gt_keep_mask=None, gt=None,mask_mult=False):
        self.r_value = r 
        self.theta_value = theta
        
        self.conf = conf
        
        self.gt_keep_mask = gt_keep_mask
        self.gt = gt
        self.mask_mult = mask_mult
        
        self.model_mean_type = conf.model_mean_type
        self.model_var_type = conf.model_var_type
        self.loss_type = conf.loss_type
        self.rescale_timesteps = conf.rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(conf.betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod -
                                                   1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) /
                                   (1.0 - self.alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas *
                                     np.sqrt(self.alphas_cumprod_prev) /
                                     (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) *
                                     np.sqrt(alphas) /
                                     (1.0 - self.alphas_cumprod))

    def training_losses(self,
                        model: Model,
                        x_start: th.Tensor,
                        x_start_post: th.Tensor,
#                         age_diff: th.Tensor,
                        t: th.Tensor,
                        model_kwargs=None,
                        noise: th.Tensor = None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {'x_t': x_t}
#         print("self.loss_type",self.loss_type)

        if self.loss_type in [
                LossType.mse,
                LossType.l1, 
                LossType.latent_code_mse
        ]:
            with autocast(self.conf.fp16):
#                 print("x_start_post_inside_base",x_start_post.shape)
                # x_t is static wrt. to the diffusion process
                model_forward = model.forward(x=x_t.detach(),
                                              t=self._scale_timesteps(t),
                                              x_start=x_start.detach(),
                                              x_start_post = x_start_post.detach(),
#                                               age_diff = age_diff.detach()
                                              **model_kwargs)
#                 model_forward_post = model.forward(x=x_t.detach(),
#                                               t=self._scale_timesteps(t),
#                                               x_start=x_start.detach(),
#                                               **model_kwargs)
            model_output = model_forward.pred
            model_cond = model_forward.cond
            model_cond_age = model_forward.cond_post
            ########################################
            gt_r_value = self.r_value 
            gt_theta_value = self.theta_value
            pred_r_value = model_forward.r_value 
            pred_theta_value = model_forward.theta_value
            #### model_r = model_forward.r = ##### model_theta = model_forward.theta = .......

            _model_output = model_output
            if self.conf.train_pred_xstart_detach:
                _model_output = _model_output.detach()
            # get the pred xstart
            p_mean_var = self.p_mean_variance(
                model=DummyModel(pred=_model_output),
                # gradient goes through x_t
                x=x_t,
                t=t,
#                 age_diff=ag_diff,
                clip_denoised=False)
            terms['pred_xstart'] = p_mean_var['pred_xstart']

            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            target_types = {
                ModelMeanType.eps: noise,
                ModelMeanType.start_x: x_start,
            }
            target = target_types[self.model_mean_type]
            #########################
#             target_r = self.r target_theta = self.theta
            assert model_output.shape == target.shape == x_start.shape

            if self.loss_type == LossType.mse:
                if self.model_mean_type == ModelMeanType.eps or self.model_mean_type == ModelMeanType.start_x :
                    # (n, c, h, w) => (n, )
                    terms["mse"] = mean_flat((target - model_output)**2)
                    ######################## 
#                     gt_r_value = self.r_value 
#                     gt_theta_value = self.theta_value
#                     pred_r_value = model_forward.r_value 
#                     pred_theta_value = model_forward.theta_value
                    terms['regres_loss'] = mean_flat((gt_r_value - pred_r_value)**2) + mean_flat((gt_theta_value - pred_theta_value)**2) 
                else:
                    raise NotImplementedError()
            #############################################################        
#             elif self.loss_type == LossType.latent_code_mse:
#                 if self.model_mean_type == ModelMeanType.eps:
#                     # (n, c, h, w) => (n, )
#                     terms["mse"] = mean_flat((target - model_output)**2)
#                     terms["latent_code_mse"] = mean_flat((model_cond[:,20::] - model_cond_post[:,20::])**2)
#                     terms["sparsity_cons"] = mean_flat( (th.cos(model_cond) + th.cos(model_cond_post)) )
#                 else:
#                     raise NotImplementedError()
            ###################################################################
            
            elif self.loss_type == LossType.l1:
                # (n, c, h, w) => (n, )
                terms["mse"] = mean_flat((target - model_output).abs())
            else:
                raise NotImplementedError()

            if "vb" in terms:
                # if learning the variance also use the vlb loss
                terms["loss"] = terms["mse"] + terms["vb"]
            elif "latent_code_mse" in terms:
                terms["loss"] = terms["mse"] + 0.2*terms["latent_code_mse"] + 0.2*terms["sparsity_cons"]
            else:
                terms["loss"] = terms["mse"] + terms['regres_loss'] 
        else:
            raise NotImplementedError(self.loss_type)

        model_cond_clone = model_cond.clone()
        model_cond_age = model_cond_age.clone()
        terms["epsilon"] = _model_output
        terms["cond"] = model_cond_clone.detach()
        terms["cond_post"] = model_cond_age.detach()
        return terms

    def sample(self,
               model: Model,
#                age_diff=None,
               shape=None,
               noise=None,
               cond=None,
               x_start=None,
               clip_denoised=True,
               model_kwargs=None,
               progress=False):
        """
        Args:
            x_start: given for the autoencoder
        """
        print("inside sample in base")
        if model_kwargs is None:
            model_kwargs = {}
            if self.conf.model_type.has_autoenc():
                model_kwargs['x_start'] = x_start
                model_kwargs['cond'] = cond

        if self.conf.gen_type == GenerativeType.ddpm:
            print("in ddpm loop from sample")
            return self.p_sample_loop(model,
                                      shape=shape,
#                                       age_diff=age_diff,
                                      noise=noise,
                                      clip_denoised=clip_denoised,
                                      model_kwargs=model_kwargs,
                                      progress=progress)
        elif self.conf.gen_type == GenerativeType.ddim:
            print("in ddim loop from sample")
            return self.ddim_sample_loop(model,
#                                          age_diff=age_diff,
                                         shape=shape,
                                         noise=noise,
                                         clip_denoised=clip_denoised,
                                         model_kwargs=model_kwargs,
                                         progress=progress)
        else:
            raise NotImplementedError()

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t,
                                        x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod,
                                            t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
            x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self,
                        model: Model,
                        x,
                        t,
#                         age_diff,
                        clip_denoised=True,
                        denoised_fn=None,
                        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B, )
        with autocast(self.conf.fp16):
            model_forward = model.forward(x=x,
                                          t=self._scale_timesteps(t),
#                                           age_diff = age_diff,
                                          **model_kwargs)
        model_output = model_forward.pred

        if self.model_var_type in [
                ModelVarType.fixed_large, ModelVarType.fixed_small
        ]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.fixed_large: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(
                        np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.fixed_small: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t,
                                                      x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type in [
                ModelMeanType.eps,
        ]:
            if self.model_mean_type == ModelMeanType.eps:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t,
                                                  eps=model_output))
            else:
                raise NotImplementedError()
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t)
        #####################################
        elif self.model_mean_type in [
                ModelMeanType.start_x,
        ]:
            pred_xstart = process_xstart(model_output)
#             print("inside model start x")
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        ################################
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'model_forward': model_forward,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
#         print("x_t.shape,eps.shape",x_t.shape,eps.shape)
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape)
            * xprev - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t)

    def _predict_xstart_from_scaled_xstart(self, t, scaled_xstart):
        return scaled_xstart * _extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, scaled_xstart.shape)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                pred_xstart) / _extract_into_tensor(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_eps_from_scaled_xstart(self, x_t, t, scaled_xstart):
        """
        Args:
            scaled_xstart: is supposed to be sqrt(alphacum) * x_0
        """
        # 1 / sqrt(1-alphabar) * (x_t - scaled xstart)
        return (x_t - scaled_xstart) / _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # scale t to be maxed out at 1000 steps
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (p_mean_var["mean"].float() +
                    p_mean_var["variance"] * gradient.float())
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(
        self,
        model: Model,
        x,
        t,
#         age_diff=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        pred_xstart=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        print("self.mask_mult",self.mask_mult)
        if self.mask_mult:

            if pred_xstart is not None:
                gt_keep_mask = self.gt_keep_mask
#                 if gt_keep_mask is None:
#                     gt_keep_mask = conf.get_inpa_mask(x)

                gt = self.gt

                alpha_cumprod = _extract_into_tensor(
                    self.alphas_cumprod, t, x.shape)

#                 if conf.inpa_inj_sched_prev_cumnoise:
#                     weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
#                 else:
                gt_weight = th.sqrt(alpha_cumprod)
                gt_part = gt_weight * gt

                noise_weight = th.sqrt((1 - alpha_cumprod))
                noise_part = noise_weight * th.randn_like(x)

                weighed_gt = gt_part + noise_part
                print("msk mult")

                x = (
                    (1 - gt_keep_mask) * (
                        weighed_gt
                    )
                    +
                    (gt_keep_mask) * (
                        x
                    )
                )
#                 x = (
#                     gt_keep_mask * (
#                         weighed_gt
#                     )
#                     +
#                     (1 - gt_keep_mask) * (
#                         x
#                     )
#                 )
#                 x_clone = x.clone()
                
#                 plt.imshow("x after",x_clone[0,0,:,:].detach().cpu())
        ####################################
        out = self.p_mean_variance(
            model,
            x,
            t,
#             age_diff=age_diff,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn,
                                              out,
                                              x,
                                              t,
                                              model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(
            0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model: Model,
#         age_diff=None,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
#                 age_diff=age_diff,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model: Model,
#         age_diff=None,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        pred_xstart=None
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = th.tensor([i] * shape[0], device=device)
            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
#                     age_diff=age_diff,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    pred_xstart=pred_xstart,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                pred_xstart = out["pred_xstart"]

    def ddim_sample(
        self,
        model: Model,
        x,
        t,
#         age_diff=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        pred_xstart=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        print("self.mask_mult",self.mask_mult)
        if self.mask_mult:

            if pred_xstart is not None:
                gt_keep_mask = self.gt_keep_mask
#                 if gt_keep_mask is None:
#                     gt_keep_mask = conf.get_inpa_mask(x)

                gt = self.gt

                alpha_cumprod = _extract_into_tensor(
                    self.alphas_cumprod, t, x.shape)

#                 if conf.inpa_inj_sched_prev_cumnoise:
#                     weighed_gt = self.get_gt_noised(gt, int(t[0].item()))
#                 else:
                gt_weight = th.sqrt(alpha_cumprod)
                gt_part = gt_weight * gt

                noise_weight = th.sqrt((1 - alpha_cumprod))
                noise_part = noise_weight * th.randn_like(x)

                weighed_gt = gt_part + noise_part
                print("msk mult")

                x = (
                    (1 - gt_keep_mask) * (
                        weighed_gt
                    )
                    +
                    (gt_keep_mask) * (
                        x
                    )
                )
#                 x = (
#                     gt_keep_mask * (
#                         weighed_gt
#                     )
#                     +
#                     (1 - gt_keep_mask) * (
#                         x
#                     )
#                 )
#                 x_clone = x.clone()
                
#                 plt.imshow("x after",x_clone[0,0,:,:].detach().cpu())
        ####################################
        
        out = self.p_mean_variance(
            model,
            x,
            t,
#             age_diff=age_diff,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn,
                                       out,
                                       x,
                                       t,
                                       model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t,
                                              x.shape)
        sigma = (eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
                 th.sqrt(1 - alpha_bar / alpha_bar_prev))
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_prev) +
                     th.sqrt(1 - alpha_bar_prev - sigma**2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        ##################################
#         from matplotlib import pyplot as plt
#         print("x")
#         plt.imshow(x[1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
#         print("sigma",th.unique(sigma))
#         print("time",t)
#         print("eps")
#         plt.imshow(eps[1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
#         print("out_pred_xstart")
#         plt.imshow(out["pred_xstart"][1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
#         print("Mean pred")
#         plt.imshow(mean_pred[1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
#         print("sample")
#         plt.imshow(sample[1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
        ####################################
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model: Model,
        x,
        t,
#         age_diff=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        NOTE: never used ? 
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
#             age_diff=age_diff,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor(
                   self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        
        ######## print
# #         print("mul_comp",_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) )
# #         print("div_comp",_extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape))
#         print("show multiplied x")
#         from matplotlib import pyplot as plt
#         plt.imshow((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x)[1,0,:,:].detach().cpu(),cmap='gray')
#         plt.show()
        ######################
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t,
                                              x.shape)

        # Equation 12. reversed  (DDIM paper)  (th.sqrt == torch.sqrt)
        mean_pred = (out["pred_xstart"] * th.sqrt(alpha_bar_next) +
                     th.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"],"eps":eps}

    def ddim_reverse_sample_loop(
        self,
        model: Model,
        x,
#         age_diff=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        device=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        sample_t = []
        xstart_t = []
        T = []
#         print("self.num_timesteps",self.num_timesteps)
        indices = list(range(self.num_timesteps))
        sample = x
        t_0 = th.randint(0,1, (sample.shape[0],)).long().to(device)
        noisy_latent = self.q_sample(x_start=sample, t=t_0, noise=None).to(device)
        sample = noisy_latent
        for i in indices:
            t = th.tensor([i] * len(sample), device=device)
            
            with th.no_grad():
                out = self.ddim_reverse_sample(model,
                                               sample,
#                                                age_diff=age_diff,
                                               t=t,
                                               clip_denoised=clip_denoised,
                                               denoised_fn=denoised_fn,
                                               model_kwargs=model_kwargs,
                                               eta=eta)
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)
#                 print(t)
#                 from matplotlib import pyplot as plt
#                 plt.imshow(sample[1,0,:,:].detach().cpu(),cmap='gray')
#                 plt.show()
#                 plt.imshow(out['pred_xstart'][1,0,:,:].detach().cpu(),cmap='gray')
#                 plt.show()
#                 plt.imshow(out['eps'][1,0,:,:].detach().cpu(),cmap='gray')
#                 plt.show()

        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }

    def ddim_sample_loop(
        self,
        model: Model,
#         age_diff=None,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
#                 age_diff=age_diff,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model: Model,
#         age_diff=None,
        shape=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        if noise is not None:
            img = noise
        else:
            assert isinstance(shape, (tuple, list))
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
#         print("self.num_timesteps",self.num_timesteps)
#         print("indices",indices)
        pred_xstart = None 
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        

        for i in indices:

            if isinstance(model_kwargs, list):
                # index dependent model kwargs
                # (T-1, ..., 0)
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = th.tensor([i] * len(img), device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
#                     age_diff=age_diff,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    pred_xstart=pred_xstart,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                yield out
                img = out["sample"]
                pred_xstart = out["pred_xstart"]

    

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size,
                      device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean,
                             logvar1=qt_log_variance,
                             mean2=0.0,
                             logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start,
                           beta_end,
                           num_diffusion_timesteps,
                           dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
        )
    elif schedule_name == "const0.01":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.01] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.015] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.008":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.008] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0065":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0065] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0055":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0055] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0045":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0045] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0035":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0035] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0025":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0025] * num_diffusion_timesteps,
                        dtype=np.float64)
    elif schedule_name == "const0.0015":
        scale = 1000 / num_diffusion_timesteps
        return np.array([scale * 0.0015] * num_diffusion_timesteps,
                        dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) +
                  ((mean1 - mean2)**2) * th.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min,
                 th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DummyModel(th.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, *args, **kwargs):
        return DummyReturn(pred=self.pred)


class DummyReturn(NamedTuple):
    pred: th.Tensor
