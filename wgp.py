import numpy as np
import torch
import gpytorch
from copy import deepcopy
from numpy.random import default_rng
import logging
import dataclasses
import typing

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
"""Warping Gaussian Process, using a known warping. Designed for synthetic data experiments, where the true warping is known ahead of time, so we can compute the shelf lives under these conditions with the shelf lives under learned-warping conditions.
"""


@dataclasses.dataclass
class WGpFitResult:
    loss: typing.Any = None
    ell: typing.Any = None
    sv: typing.Any = None
    best_index: int = None
    offset: typing.Any = None


def _format_tuple(tup, fmt):
    return " ".join(format(s, fmt) for s in tup)


def interp(x, xp, fp):
    """One-dimensional linear interpolation using torch functions.
    From https://github.com/pytorch/pytorch/issues/50334

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the function, must be increasing.
        fp: the :math:`y`-coordinates of the function, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.searchsorted(xp, x, right=False) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


class WGPModel(gpytorch.models.ExactGP):
    """A nonstationary GP model that uses a known warping."""

    def __init__(
        self,
        train_inputs,
        train_targets,
        x_grid,
        warp_x_grid,
        measurement_noise_var,
        signal_var=None,
        ell=None,
        seed=None,
    ):
        """Initializes the model"""
        train_inputs = torch.as_tensor(train_inputs).float()

        train_targets = torch.as_tensor(train_targets).float()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = measurement_noise_var
        likelihood.noise_covar.raw_noise.requires_grad_(False)

        super().__init__(train_inputs, train_targets, likelihood)
        self.orig_inputs = train_inputs
        self.orig_targets = train_targets
        self.x_grid = torch.as_tensor(x_grid).float()
        self.warp_x_grid = torch.as_tensor(warp_x_grid).float()

        dw = torch.diff(self.warp_x_grid)
        dx = torch.diff(self.x_grid)
        self.x_grid_volatility = dw / dx

        self.rng = default_rng(seed)
        self.ks_ = []

        self.measurement_noise_var = measurement_noise_var

        est_mean, est_var = self._estimate_mean_var(
            self.train_inputs[0], self.train_targets
        )
        if signal_var is not None:
            est_var = signal_var

        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = est_mean
        self.mean_module.raw_constant.requires_grad = False

        if ell is None:
            ell = 1.0
        # May require bounds, or a prior
        k0 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(),
        )
        k0.base_kernel.lengthscale = ell
        k0.base_kernel.raw_lengthscale.requires_grad = True

        # Fixing the outputscale to the estimate forces all of the learning to
        # be captured by the lengthscale.
        k0.outputscale = est_var
        k0.raw_outputscale.requires_grad = False

        self.ks_.append(k0)
        self.covar_module = k0

    @property
    def lengthscales(self):
        return [k.base_kernel.lengthscale.item() for k in self.ks_]

    @lengthscales.setter
    def lengthscales(self, value):
        for k, v in zip(self.ks_, value):
            k.lengthscale = torch.tensor([v])

    @property
    def loss(self):
        return self.best_loss.item()

    @property
    def signal_vars(self):
        return [k.outputscale.item() for k in self.ks_]

    @property
    def offset(self):
        return self.mean_module.constant.item()

    def warp(self, x):
        return interp(
            torch.squeeze(x, dim=-1),
            torch.squeeze(self.x_grid, dim=-1),
            self.warp_x_grid,
        )

    def volatility(self, x):
        return interp(
            torch.squeeze(x, dim=-1),
            torch.squeeze(self.x_grid[1:], dim=-1),
            self.x_grid_volatility,
        )

    def _estimate_mean_var(self, x, y, n_pts=50):
        """Estimate the mean and variance of the function through points (x, y)

        Works by fitting a linear interpolation through the points, then finding the mean and variance of that curve. This is probably a little more accurate than just working with the points themselves.
        Args:
            x, y (float): points sampled from an underlying 2d function y = f(x).
            n_pts: Number of points to use for function grid.
        Returns:
            float: The estimated mean and variance of f, in the Gaussian Process sense.
        """
        x_grid = torch.linspace(x[0].item(), x[-1].item(), n_pts)
        y_grid = interp(x_grid, torch.squeeze(x), y)
        return torch.mean(y_grid), torch.var(y_grid)

    def set_train_data(self, inputs, targets, strict):
        # We set the training data here to get forward-looking uncertainties, but we do not re-set the warping.
        super().set_train_data(
            torch.as_tensor(inputs).float(),
            torch.as_tensor(targets).float(),
            strict,
        )

        # We must also adjust the noise vector for a FixedNoiseGaussianLikelihood. Here we assume that the measurement noise is constant across all inputs. If we don't do this, any call of the model will raise a very misleading error (misleading because the call to model(test_x) does not explicitly pass data through the likelihood):
        # GPInputWarning: You have passed data through a FixedNoiseGaussianLikelihood that did not match the size of the fixed noise, *and* you did not specify noise. This is treated as a no-op.

        # self.likelihood.noise = (
        #     torch.ones_like(self.train_targets) * self.measurement_noise_var
        # )

    def forward(self, x):
        wx = self.warp(x)
        mean_x = self.mean_module(wx)
        covar_x = self.covar_module(wx)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(
        self,
        *,
        n_iter=100,
        progress_threshold=100,
        smooth_weight=1.0,
        accel_sd=3.0,
        tol=0.0,
        lr=0.1,
        verbose=False,
    ):
        """Iteratively fit the model, with stopping criteria.

        Stops after a run of iterations where improvement in marginal log likelihood is less than `progress_threshold` for each iteration,  or after `n_iter` total iterations, whichever comes first.

        Args:
            n_iter (int, optional): Maximum number of fitting iterations.Defaults to 100.
            progress_threshold (int, optional): Number of iterations to continue if progress below `tol` is made. Defaults to 100.
            tol (float, optional): Minimum change in marginal log likelihood to count as progress. Defaults to 0.0.
            lr (float, optional): Optimizer learning rate. Defaults to 0.1.
            verbose (bool, optional): Use verbose logging. Defaults to False.

        Returns:
            _type_: _description_
        """
        # self.mean_module.constant = self.rng.uniform(-10.0, 10.0)
        # self.mean_module.raw_constant.requires_grad = True

        losses = np.empty(n_iter)
        offsets = np.empty(n_iter)
        n_kernels = len(self.ks_)
        lengthscales = np.empty((n_iter, n_kernels))
        signal_vars = np.empty((n_iter, n_kernels))
        warped = np.empty(
            (n_iter, len(self.train_targets)),
        )
        self.best_loss = np.inf
        last_loss = np.inf
        best_index = None

        self.train()
        self.likelihood.train()

        params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.likelihood, self)
        accel_prior = gpytorch.priors.NormalPrior(0.0, accel_sd)
        n_small_improvement = 0
        wx = self.warp(self.train_inputs[0])

        for i in range(n_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.__call__(self.train_inputs[0])

            # Calc loss and backprop
            # gradients
            loss = -mll(output, self.train_targets)
            loss.backward()

            losses[i] = loss
            lengthscales[i, :] = self.lengthscales
            signal_vars[i, :] = self.signal_vars
            offsets[i] = self.offset

            best_flag = ""
            n_small_improvement += 1

            if last_loss - loss > tol:
                n_small_improvement = 0

            if loss < self.best_loss:
                best_flag = "*"
                self.best_loss = loss
                best_model = deepcopy(self.state_dict())
                best_index = i

            log.debug(f"Iter {i:04d}/{n_iter} {self} ({best_index:04d}) {best_flag}")

            if n_small_improvement > progress_threshold:
                log.debug(
                    f"Stopped after {n_small_improvement} iterations with only small improvement."
                )
                break
            last_loss = loss
            optimizer.step()

        self.load_state_dict(best_model)
        j = i + 1
        return WGpFitResult(
            loss=losses[:j],
            ell=lengthscales[:j, :],
            sv=signal_vars[:j, :],
            best_index=best_index,
            offset=offsets[:j],
        )

    def predict(self, test_inputs):
        self.eval()
        self.likelihood.eval()
        test_inputs = torch.as_tensor(test_inputs).float()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            a = self.__call__(test_inputs)
            preds = self.likelihood(a)

        p_mean = preds.mean.numpy()
        p_lower, p_upper = [x.detach().numpy() for x in preds.confidence_region()]
        return p_mean, p_lower, p_upper

    def _uncertainty_at(self, x, obs_x, obs_y):
        """Compute uncertainty at locations x given the observations.

        Uses current hyperparameters, adjusting posterior using only obs_x and obs_y.

        Args:
            x (vector of float): locations at which to compute uncertainty
            obs_x, obs_y (vector of float): train_inputs and train_targets to use.

        Returns:
            (mean, lower, upper) values at locations x.
        """
        self.set_train_data(inputs=obs_x, targets=obs_y, strict=False)
        p_mean, p_lower, p_upper = self.predict(x)

        self.set_train_data(
            inputs=self.orig_inputs, targets=self.orig_targets, strict=False
        )

        return (p_mean, p_lower, p_upper)

    def monitoring_uncertainty(self):
        """Compute monitoring uncertainty at all inputs.

        The monitoring uncertainty of the first input is always larger than the others, because no observations have been made yet.

        Monitoring uncertainty at input location t is the width of the confidence interval when observations x[i] < t are given. This value is computed iteratively for each t=x[i], given values for x[:i].

        Returns:
            List of (mean, lower, upper) vectors of confidence limits.
        """

        # TODO: Consider for future optimization replacing FixedNoiseGaussianLiklihood with GaussianLikelihoodWithMissingObs, and then instead of re-setting the training data, set the values of train_targets to NaN incrementally, starting from the far end. This might be convertable to a matrix version, which could find all of the measurement uncertainties for a single curve in one shot, instead of one at a time, like we do here.

        x = self.train_inputs[0].detach().squeeze().numpy()
        y = self.train_targets.detach().numpy()

        n = len(x)
        p_mean = np.empty(n)
        p_lower = np.empty(n)
        p_upper = np.empty(n)

        for i in range(n):
            trunc_x = x[:i]
            trunc_y = y[:i]
            next_x = x[i]

            e_mean, e_lower, e_upper = self._uncertainty_at((next_x,), trunc_x, trunc_y)

            p_mean[i] = e_mean[0]
            p_lower[i] = e_lower[0]
            p_upper[i] = e_upper[0]

        return p_mean, p_lower, p_upper

    def monitoring_curves(self, num=10):
        """Compute monitoring uncertainty curves.

        Monitoring curves are the uncertainties that grow from x[n] to  x[n+1], given all observations < x[n].

        Args:
            num (int, optional): Number of curve points to calculate between each x[i].

        Returns:
            List of (curve_x, lower, upper) vectors of curves.
        """

        # TODO: Consider for future optimization replacing FixedNoiseGaussianLiklihood with GaussianLikelihoodWithMissingObs, and then instead of re-setting the training data, set the values of train_targets to NaN incrementally, starting from the far end. This might be convertable to a matrix version, which could find all of the measurement uncertainties for a single curve in one shot, instead of one at a time, like we do here.

        x = self.train_inputs[0].detach().squeeze(dim=1).numpy()
        y = self.train_targets.detach().numpy()
        n = len(x) - 1
        p_x = []
        p_lower = []
        p_upper = []

        for i in range(n):
            trunc_x = x[: i + 1]
            trunc_y = y[: i + 1]
            next_x = x[i + 1]
            curve_x = np.linspace(trunc_x[-1], next_x, num)

            _, e_lower, e_upper = self._uncertainty_at(curve_x, trunc_x, trunc_y)

            p_x.append(curve_x)
            p_lower.append(e_lower)
            p_upper.append(e_upper)

        return p_x, p_lower, p_upper

    def shelf_life(self, n, target_uncertainty, max_life=None, num_pts=50):
        """Compute shelf life of data point n.

        Computes the time until the uncertainty reaches target_uncertainty after observation n, given all previous observations. Uses current hyperparameters and current volatility to recompute the GP given the observations.

        Any shelf life beyond max_life is set to max_life, and marked as censored. Although if max_life == 0, NaN is returned, and the points are marked as censored.

        If target_uncertainty < sqrt(self.noise_var), returns 0. If time is more than max_life, returns max_life.

        Args:
            n (int): observation index
            target_uncertianty (iterable of float): the desired full width uncertainty, in units of self.train_targets.
            num_pts (int, optional): number of points to use in the shelf life extrapolation. More points means a more accurate determination. Defaults to 30.
            max_life (float, optional): the maximum shelf life computed, in units of self.train_inputs. Default is the time between the nth observation and the final observation.

        Returns:
            (array of float) time, in units of self.train_inputs, after self.train_inputs[n] at which the full uncertainty reaches target_uncertainty.
            (array of bool) True if the shelf life calculation was complete (not censored)
        """
        x = self.train_inputs[0].detach().squeeze(dim=1).numpy()
        y = self.train_targets.detach().numpy()
        target_uncertainty = np.atleast_1d(target_uncertainty)
        life = np.empty_like(target_uncertainty)
        complete = np.full_like(target_uncertainty, True, dtype=bool)

        if max_life is None:
            max_life = x[-1] - x[n]

        if max_life == 0.0:
            life[:] = np.nan
            complete[:] = False
            return life, complete

        # Logarithmically spaced points in the curve, because the uncertainty always changes faster at the beginning, and doing it this way makes for a little more accurate interpolation later.
        # 0.01 days is about 15 minutes, but not really the first bin size, because of the manipulation we do next to get things to start at 0.
        start_time = 0.01
        curve_x = (
            np.geomspace(start_time, max_life + start_time, num_pts + 1) - start_time
        )
        curve_x += x[n]

        p_lower = np.empty(num_pts)
        p_upper = np.empty(num_pts)

        trunc_x = x[: n + 1]
        trunc_y = y[: n + 1]

        _, p_lower, p_upper = self._uncertainty_at(curve_x, trunc_x, trunc_y)
        # if curve_x starts at the last given observation, unc should be nondecreasing, so this search should work.
        unc = p_upper - p_lower
        k = np.searchsorted(unc, target_uncertainty)

        # linear interpolation to compute a more accurate shelf life.
        # Shelf life is relative to x[n], which is curve_x[0]. If we were working with scalar target_uncertainties, it would look like a simple interpolation:
        #     frac = (target_uncertainty - unc[k - 1]) / (unc[k] - unc[k - 1])
        #     life = curve_x[k - 1] + frac * (curve_x[k] - curve_x[k - 1]) - curve_x[0]
        # But because we want to do this for multiple target_uncertainties at once, we have to use indexing to handle the special cases of zero and max elements.

        max_elts = k >= len(unc)

        # There shouldn't be too many of these, because it means target_uncertainty was smaller than measurement error.
        zero_elts = k == 0

        mask = np.logical_not(np.logical_or(max_elts, zero_elts))
        km = k[mask]

        life[max_elts] = max_life
        complete[max_elts] = False

        life[zero_elts] = 0

        frac = (target_uncertainty[mask] - unc[km - 1]) / (unc[km] - unc[km - 1])
        life[mask] = (
            curve_x[km - 1] + frac * (curve_x[km] - curve_x[km - 1]) - curve_x[0]
        )

        return life, complete

    def all_shelf_life(self, target_uncertainty, max_life=None, num_pts=50):
        """Compute shelf life of all data points.

        For each observation, computes the time until the uncertainty reaches target_uncertainty after the observation, given all previous observations. Uses current hyperparameters and current volatility to recompute the GP given the observations. If target_uncertainty < sqrt(self.noise_var), returns 0. If time is more than max_life, returns max_life.

        Args:
            n (int): observation index
            target_uncertianty (iterable of float): the desired full width uncertainty, in units of self.train_targets.
            num_pts (int, optional): number of points to use in the shelf life extrapolation. More points means a more accurate determination. Defaults to 30.
            max_life (float, optional): the maximum shelf life computed, in units of self.train_inputs. Defaults to the time between the indexed observation and the last observation.

        Returns:
            (ndarray of float) time, in units of self.train_inputs, after self.train_inputs[n] at which the full uncertainty reaches target_uncertainty. One row per element of target_uncertainty.
        """
        shelf_life = []
        complete = []

        for i in range(len(self.train_inputs[0].detach().squeeze(dim=1).numpy())):
            l, c = self.shelf_life(
                i,
                target_uncertainty=target_uncertainty,
                max_life=max_life,
                num_pts=num_pts,
            )
            shelf_life.append(l)
            complete.append(c)
        return np.asarray(shelf_life).T, np.asarray(complete).T

    def __str__(self):
        return f"LWGPModel Loss: {self.loss:0.4f}, ls: {_format_tuple(self.lengthscales, '0.2f')}, sv: {_format_tuple(self.signal_vars, '0.2f')}"
        # return f"GPModel Loss: {self.loss:0.4f}, ells: {*self.lengthscales,:0.4f}, svar: {*self.signal_vars,:0.4f}"
