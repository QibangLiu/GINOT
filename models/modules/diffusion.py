# %%
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math
import numpy as np
from tqdm import tqdm

# %%
"""Classifier free guidance Diffusion Model"""


def beta_scheduler(tot_timesteps, type="linear", s=0.008):
    if type == "linear":
        scale = 1000 / tot_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, tot_timesteps)
    elif type == "sigmoid":
        betas = torch.linspace(-6, 6, tot_timesteps)
        betas = (
            torch.sigmoid(betas)
            / (betas.max() - betas.min())
            * (0.02 - betas.min())
            / 10
        )
        return betas
    elif type == "cosine":
        """
        cosine schedule
        as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = tot_timesteps + 1
        x = torch.linspace(0, tot_timesteps, steps, dtype=torch.float64)
        alphas_cumprod = (
            torch.cos(((x / tot_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"unknown beta scheduler {type}")


class GaussianDiffusion:
    def __init__(self, img_shape, timesteps=1000, beta_schedule="linear"):
        self.img_shape = img_shape
        self.timesteps = timesteps

        self.betas = beta_scheduler(timesteps, type=beta_schedule)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = nnF.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(
            1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2],
                      self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        """
        t is time step in [0, total_time_step], shape: (batch_size, )
        take the value of a at t
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion (using the nice property): q(x_t | x_0)
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
        t: (batch_size, )
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        # (total_timesteps), (nb, ) -> (nb, 1,...)
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        """
        q(x_t | x_0);
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise;
        mu=sqrt(\bar{alpha_t}) * x_0;
        sigma^2=1-\bar{alpha_t};
        """
        mean = self._extract(self.sqrt_alphas_cumprod,
                             t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        diffusion posterior: q(x_{t-1} | x_t, x_0)
        mu=coe1 * x_0 + coe2 * x_t;
        coe1=beta_t * sqrt(\bar{\alpha_{t-1}}) / (1 - \bar{\alpha_t});
        coe2=(1 - \bar{\alpha_{t-1}}) * sqrt(\bar{\alpha_t}) / (1 - \bar{\alpha_t});
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #
    def predict_start_from_noise(self, x_t, t, noise):
        """
        compute x_0 from x_t and noise
        reverse of `q_sample`:
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod,
                            t, x_t.shape) * noise
        )

    #
    def p_mean_variance(
        self, model, x_t, t, c, w, clip_denoised=True, conditioning=True
    ):
        """
        compute predicted mean and variance of p(x_{t-1} | x_t)

        noise is from the model

        x0 is from the inverse of the q_sample, q(x_t | x_0) -->x0,
        x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise


        then diffusion posterior: q(x_{t-1} | x_t, x_0) to get the mean and variance

        """
        device = next(model.parameters()).device
        batch_size = x_t.shape[0]
        # predict noise using model

        pred_noise_none = model(
            x_t, t, c, torch.zeros(batch_size).int().to(device))
        if conditioning:
            pred_noise_c = model(x_t, t, c, torch.ones(
                batch_size).int().to(device))
            pred_noise = (1 + w) * pred_noise_c - w * pred_noise_none
        else:
            pred_noise = pred_noise_none

        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(x_recon, x_t, t)
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, x_t, t, c, w, clip_denoised=True, conditioning=True):
        """
        denoise_step: sample x_{t-1} from x_t and pred_noise:
        1. compute predicted mean and variance of p(x_{t-1} | x_t):
            a. noise is from the model
            b. x0 is from the inverse of the q_sample, q(x_t | x_0) -->x0,
               x_t= sqrt(\bar{alpha_t}) * x_0 + sqrt(1 - \bar{alpha_t}) * noise
            c. then diffusion posterior: q(x_{t-1} | x_t, x_0) to get the mean and variance
        2. sample x_{t-1} from the predicted mean and variance
        """
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, c, w, clip_denoised=clip_denoised, conditioning=conditioning
        )
        noise = torch.randn_like(x_t)
        # no noise when t == 0 QB: to be look into???
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * \
            (0.5 * model_log_variance).exp() * noise
        return pred_img

    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        image_shape,
        labels,
        w=2,
        clip_denoised=True,
        conditioning=True,
        timesteps=None,
        all_timesteps=False,
    ):
        if timesteps is None:
            timesteps = self.timesteps

        batch_size = labels.shape[0]
        device = next(model.parameters()).device
        labels = labels.to(device)
        # start from pure noise (for each example in the batch)
        img = torch.randn((batch_size, *image_shape), device=device)
        imgs = []
        for i in tqdm(
            reversed(range(0, timesteps)),
            desc="sampling loop time step",
            total=timesteps,
        ):
            img = self.p_sample(
                model,
                img,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                labels,
                w,
                clip_denoised,
                conditioning=conditioning,
            )
            if all_timesteps:
                imgs.append(img.cpu().numpy())

        if all_timesteps:
            return np.array(imgs)
        else:
            return img.cpu().numpy()

    # sample new images

    @torch.no_grad()
    def sample(
        self,
        model,
        labels,
        w=2,
        image_shape=None,
        clip_denoised=True,
        conditioning=True,
        timesteps=None,
        all_timesteps=False,
    ):
        if image_shape is None:
            image_shape = self.img_shape
        return self.p_sample_loop(
            model,
            image_shape,
            labels,
            w,
            clip_denoised,
            conditioning,
            timesteps=timesteps,
            all_timesteps=all_timesteps
        )

    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,

        labels,
        image_shape=None,
        ddim_timesteps=50,
        w=2,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        conditioning=True,
    ):
        if image_shape is None:
            image_shape = self.img_shape
        # make ddim timestep sequence
        if ddim_discr_method == "uniform":
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == "quad":
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.timesteps * 0.8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{ddim_discr_method}"'
            )
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(
            np.array([0]), ddim_timestep_seq[:-1])

        device = next(model.parameters()).device

        labels = labels.to(device)
        batch_size = labels.shape[0]
        # start from pure noise (for each example in the batch)
        sample_img = torch.randn((batch_size, *image_shape), device=device)
        seq_img = [sample_img.cpu().numpy()]

        for i in tqdm(
            reversed(range(0, ddim_timesteps)),
            desc="sampling loop time step",
            total=ddim_timesteps,
        ):
            t = torch.full(
                (batch_size,
                 ), ddim_timestep_seq[i], device=device, dtype=torch.long
            )
            prev_t = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long,
            )

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(
                self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(
                self.alphas_cumprod, prev_t, sample_img.shape
            )

            # 2. predict noise using model

            pred_noise_none = model(
                sample_img, t, labels, torch.zeros(
                    batch_size, dtype=torch.int32).to(device)
            )
            if conditioning:
                pred_noise_c = model(
                    sample_img, t, labels, torch.ones(
                        batch_size, dtype=torch.int32).to(device)
                )
                pred_noise = (1 + w) * pred_noise_c - w * pred_noise_none
            else:
                pred_noise = pred_noise_none
            # 3. get the predicted x_0
            pred_x0 = (
                sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise
            ) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            )

            # 6. compute x_{t-1} of formula (12)
            x_prev = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir_xt
                + sigmas_t * torch.randn_like(sample_img)
            )

            sample_img = x_prev

        return sample_img.cpu().numpy()

    # compute train losses
    def train_losses(self, model, x_start, t, c, mask_c):
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t, c, mask_c)
        loss = nnF.mse_loss(noise, predicted_noise)
        return loss
