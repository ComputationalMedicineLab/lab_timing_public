import numpy as np
import pandas as pd
import torch
import gpytorch
import pickle
from copy import deepcopy
from matplotlib import pyplot as plt
from numpy.random import default_rng
from timeit import default_timer as timer
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import PchipInterpolator, interp1d
from datetime import datetime
import logging
import pickle
from tqdm.notebook import trange, tqdm
import itertools

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
"""The up-to-date file for Nonstationary GP code.  But still rather messy.
"""


def _format_tuple(tup, fmt):
    return " ".join(format(s, fmt) for s in tup)


class Warper:
    def __init__(self):
        self.x = None
        self.y = None
        self.ells = None
        self.dx = None
        self.w = None
        self._warp_fun = None
        self._volatility_fun = None

    def _setup(self, x, y, ells):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.ells = ells

        self.dx = np.diff(x)
        self.w = np.concatenate(([0], np.cumsum(self.dx / self.ells)))

    def create_warp_fun(self, x, y, ells):
        raise NotImplementedError()

    def create_volatility_fun(self):
        raise NotImplementedError()


class PchipWarper(Warper):
    def __init__(self):
        super().__init__()

    def create_warp_fun(self, x, y, ells):
        self._setup(x, y, ells)

        # Extend w and x by repeating the first and last intervals at their same
        # slopes, in order to force pchip to use better slopes at the original
        # end points. This implementation of phcip sets them at zero otherwise.
        x_extended = np.concatenate(([x[0] - self.dx[0]], x, [x[-1] + self.dx[-1]]))

        dw = np.diff(self.w)
        w_extended = np.concatenate(
            ([self.w[0] - dw[0]], self.w, self.w[-1] + [dw[-1]])
        )
        self._warp_fun = PchipInterpolator(x_extended, w_extended)
        return self._warp_fun

    def create_volatility_fun(self):
        return self._warp_fun.derivative()


class PiecewiseLinearWarper(Warper):
    def __init__(self):
        super().__init__()

    def create_warp_fun(self, x, y, ells):
        self._setup(x, y, ells)

        def helper(x):
            return np.interp(x, self.x, self.w)

        return helper

    def create_volatility_fun(self):
        # Repeat the end elements of ells, for extrapolating beyond the boundaries of x. That shouldn't happen often, but we don't want to crash if it does. This simply uses the first and last lengthscales to anything beyond them.
        self._ells_extended = 1.0 / np.concatenate(
            ([self.ells[0]], self.ells, [self.ells[-1]])
        )

        def helper(x):
            return self._ells_extended[np.searchsorted(self.x, x)]

        return helper


class GPModel(gpytorch.models.ExactGP):
    """A stationary GP model with certain assumptions built in.

    Assumes a covariance function composed of:
      - a short lengthscale squared exponential
      - a long lengthscale squared exponential
      - a constant mean function
      - an amplitude parameter for the squared exponentials
      - a fixed measurement noise parameter
    """

    def __init__(
        self,
        train_inputs,
        train_targets,
        measurement_noise_var,
        seed=None,
    ):
        """Initializes the model

        Args:
            train_inputs (iterable of float): observation locations
            train_targets (iterable of float): observation values
            measurement_noise_var (float): fixed variance of measurement noise.
            seed (integer): optional random seed. If None (default), uses the system value, which will be different for each run.
        """

        train_inputs = torch.as_tensor(train_inputs).float()
        train_targets = torch.as_tensor(train_targets).float()
        noises = torch.ones_like(train_targets) * measurement_noise_var
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
        super().__init__(train_inputs, train_targets, likelihood)

        self.rng = default_rng(seed)
        self.ks_ = [None, None]
        self.measurement_noise_var = measurement_noise_var

        self.mean_module = gpytorch.means.ConstantMean()

        # short lengthscale kernel
        k0 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # initialize to the smallest time delta between two training points.
        k0.base_kernel.lengthscale = torch.min(torch.diff(train_inputs))
        k0.base_kernel.raw_lengthscale.requires_grad = True
        k0.raw_outputscale.requires_grad = True
        self.ks_[0] = k0

        # longer lengthscale kernel
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # initialize to the length of the training data
        k1.base_kernel.lengthscale = train_inputs[-1] - train_inputs[0]
        k1.base_kernel.raw_lengthscale.requires_grad = True
        k1.raw_outputscale.requires_grad = True
        self.ks_[1] = k1

        # Final covariance function is the sum of the short and long lengthscale kernels.
        # unfortunately, sum(self.ks_) won't work, so we can't make this general for an arbitrary number of kernels.
        self.covar_module = k0 + k1

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

    def set_train_data(self, inputs, targets, strict):
        super().set_train_data(
            torch.as_tensor(inputs).float(),
            torch.as_tensor(targets).float(),
            strict,
        )

        # We must also adjust the noise vector for a FixedNoiseGaussianLikelihood. Here we assume that the measurement noise is constant across all inputs. If we don't do this, any call of the model will raise a very misleading error (misleading because the call to model(test_x) does not explicitly pass data through the likelihood):
        # GPInputWarning: You have passed data through a FixedNoiseGaussianLikelihood that did not match the size of the fixed noise, *and* you did not specify noise. This is treated as a no-op.

        self.likelihood.noise = (
            torch.ones_like(self.train_targets) * self.measurement_noise_var
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(
        self,
        *,
        n_iter=100,
        progress_threshold=100,
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
        self.mean_module.constant = self.rng.uniform(-10.0, 10.0)
        self.mean_module.raw_constant.requires_grad = True

        losses = np.empty(n_iter)
        n_kernels = len(self.ks_)
        lengthscales = np.empty((n_iter, n_kernels))
        signal_vars = np.empty((n_iter, n_kernels))
        self.best_loss = np.inf
        best_index = None
        last_loss = np.inf
        self.train()
        self.likelihood.train()

        params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        n_no_improvement = 0
        n_small_improvement = 0

        for i in range(n_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.__call__(self.train_inputs[0])
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_targets)
            loss.backward()

            losses[i] = loss
            lengthscales[i, :] = self.lengthscales
            signal_vars[i, :] = self.signal_vars

            best_flag = ""
            n_no_improvement += 1
            n_small_improvement += 1
            if loss < self.best_loss:
                n_no_improvement = 0
                best_flag = "*"
                self.best_loss = loss
                best_model = deepcopy(self.state_dict())
                best_index = i

            log.debug(f"Iter {i:04d}/{n_iter} {self} ({best_index:04d}) {best_flag}")

            if n_no_improvement > progress_threshold:
                log.debug(
                    f"Stopped after {n_no_improvement} iterations without improvement."
                )
                break

            if last_loss - loss > tol:
                n_small_improvement = 0

            if n_small_improvement > progress_threshold:
                log.debug(
                    f"Stopped after {n_small_improvement} iterations with only small improvement."
                )
                break
            last_loss = loss
            optimizer.step()

        self.load_state_dict(best_model)
        j = i + 1
        return (
            losses[:j],
            lengthscales[:j, :],
            signal_vars[:j, :],
            best_index,
        )

    def predict(self, test_inputs):
        self.eval()
        self.likelihood.eval()
        test_inputs = torch.as_tensor(test_inputs).float()
        noises = torch.ones_like(test_inputs) * self.measurement_noise_var
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            a = self.__call__(test_inputs)
            preds = self.likelihood(a, noise=noises)

        p_mean = preds.mean.numpy()
        p_lower, p_upper = [x.detach().numpy() for x in preds.confidence_region()]
        return p_mean, p_lower, p_upper

    def __str__(self):
        return f"GPModel Loss: {self.loss:0.4f}, ls: {_format_tuple(self.lengthscales, '0.2f')}, sv: {_format_tuple(self.signal_vars, '0.2f')}"
        # return f"GPModel Loss: {self.loss:0.4f}, ells: {*self.lengthscales,:0.4f}, svar: {*self.signal_vars,:0.4f}"


class NSGPModel:
    def __init__(
        self,
        train_inputs,
        train_targets,
        noise_var,
        ell_max,
        ell_min,
        ell_func,
        warper=PchipWarper(),
    ):
        """Nonstationary GP Model.

        Model assumes fixed measurement noise and 1-dimensional inputs (such as time or 1-d location.)

        An interpolating lookup table is needed for NSGPModel. This table must be provided so the model can look up the best Gaussian Process length scale for two points $D_1 = (x_1, y_1)$ and $D_2 = (x_2, y_2)$. The lookup table $f$ is constructed such that what we look up is $\log(ell) = f(n, s),$ where $n$ indicates normalized measurement noise level and $s$ indicates normalized signal level. The normalization is such that $D_1 = (0,0)$ and $D_2 = (1,1)$. 

        We compute $n$ and $s$ from $\sigma_n$ and $\sigma_s$. We fix $\sigma_n$ by what we know about the lab test measurement noise (standard deviation of the measurement error), and we estimate $\sigma_s$ as the standard deviation of the signal, by taking the standard devation of a quick smooth fit to the measurements. That won't be a perfect estimate, but it's the best we have. 

        If  $\Delta y = y_2 - y_1$, $\Delta x = x_2 - x_1$, then we 
        have $n = \log(\sigma_n / \Delta y)$ and $s = \log(\sigma_s / \Delta y).$

        Then we compute 
        \begin{align}
            ell &= 10^{f(n, s)} \Delta x \\
                &= 10^{f(\log(\sigma_n / \Delta y), \log(\sigma_s / \Delta y)} \Delta x.
        \end{align}  

        Args:
            train_inputs (iterable of float): observation times (x)
            train_targets (iterable of float): observation values (y)
            noise_var (iterable of float): variance of train_targets measurement noise. (stdev**2)
            ell_max (float): maximum lengthscale allowed (units of x)
            ell_min (float): minimum lengthscale allowed
            ell_func (interpolate.interp2d): lengthscale lookup function meeting the criteria described above.
            warper (Warper()): Creates the warping function. Defaults to PchipWarper().
             
        """
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.noise_var = noise_var
        self.ell_func = ell_func
        self.ell_max = ell_max
        self.ell_min = ell_min
        self.warper = warper
        self.ells_ = None
        self.signal_std_ = None
        self.volatility_fun_ = None
        self.warp_fun_ = None

        # The internal, stationary gp trained on warped inputs
        self.gp_ = None
        self._patch_axis = None

    def _get_stdf(self, x, y, n_pts=50):
        """Estimate the standard devation of the function through points (x, y)

        Works by fitting a pchip interpolation through the points, then finding the standard devation of that curve. This is probably a little more accurate than just taking the standard devation of the points themselves, unless those points are equally spaced through their range.
        Args:
            x, y (float): points sampled from an underlying 2d function y = f(x).
            n_pts: Number of points to use for function grid.
        Returns:
            float: The estimated standard devation of f, in the Gaussian Process sense.
        """
        y_grid = np.linspace(x[0], x[-1], n_pts)
        y_grid = PchipInterpolator(x, y, extrapolate=False)(y_grid)
        return np.std(y_grid, ddof=0)

    def _compute_two_point_length_scales(self, dx, dy, signal_std, noise_std):
        adjusted_ell_min = self.ell_min / dx
        adjusted_ell_max = self.ell_max / dx

        # Add noise in quadrature to the differences. This treats identical numbers as if they were different by sqrt_2 * noise_std. Because identical numbers produce an infinite length scale, and they are only identical by accident, after the additive measurement noise. A bit of a hack, but better than leaving them identical, and easier than treating everything as a probability distribution.
        ady = np.sqrt(np.power(dy, 2) + 2 * noise_std**2)
        n = np.log10(signal_std / ady)
        s = np.log10(noise_std / ady)
        log_norm_ell = self.ell_func(n, s, grid=False)
        ell = np.maximum(
            adjusted_ell_min, np.minimum(adjusted_ell_max, np.power(10, log_norm_ell))
        )
        return ell * dx

    def _estimate_length_scales(self):
        x = self.train_inputs
        y = self.train_targets
        sig_eps = 1.0e-2 * self.noise_var
        self.signal_std_ = self._get_stdf(x, y) + sig_eps
        self.ells_ = self._compute_two_point_length_scales(
            np.diff(x), np.diff(y), self.signal_std_, np.sqrt(self.noise_var)
        )
        return self.ells_

    def _estimate_warp_fun(self, num=100):
        x = self.train_inputs
        y = self.train_targets
        self._estimate_length_scales()
        self.warp_fun_ = self.warper.create_warp_fun(x, y, self.ells_)

        # dx = np.diff(x)

        # # Extend w and x by repeating the first and last intervals at their same
        # # slopes, in order to force pchip to use better slopes at the original
        # # end points. This implementation of phcip sets them at zero otherwise.
        # x_extended = np.concatenate(([x[0] - dx[0]], x, [x[-1] + dx[-1]]))
        # w = np.concatenate(([0], np.cumsum(dx / self.ells_)))
        # dw = np.diff(w)
        # w_extended = np.concatenate(([w[0] - dw[0]], w, w[-1] + [dw[-1]]))
        # self.warp_fun_ = PchipInterpolator(x_extended, w_extended)
        # self.warp_fun_ = interp1d(x_extended, w_extended, fill_value="extrapolate")
        # alternate warping approach - linear interpolation. Would require
        # fixing the volatility function to piecewise constant.
        # self.warp_fun_ = interp1d(x, w)

    def _estimate_volatility_fun(self, num=100):
        # Assumes that the volatility function depends on the warp function. May have to revisit that assumption if we try a method that is the other way around.
        if self.warp_fun_ is None:
            self._estimate_warp_fun(num=num)
        self.volatility_fun_ = self.warper.create_volatility_fun()

    def warp(self, x):
        if self.warp_fun_ is None:
            self._estimate_warp_fun()
        return self.warp_fun_(x)

    def volatility(self, x):
        if self.volatility_fun_ is None:
            self._estimate_volatility_fun()
        return self.volatility_fun_(x)

    @property
    def warped_inputs(self):
        return self.warp(self.train_inputs)

    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.gp_.set_train_data(
            self.warp(inputs),
            targets,
            strict,
        )

    def forward(self, x):
        return self.gp_.forward(x)

    def fit(self, **kwargs):
        self.gp_ = GPModel(
            self.warped_inputs,
            self.train_targets,
            measurement_noise_var=self.noise_var,
        )
        return self.gp_.fit(**kwargs)

    def predict(self, x):
        return self.gp_.predict(self.warp(x))

    def _uncertainty_at(self, x, obs_x, obs_y):
        """Compute uncertainty at locations x given the observations.

        Uses current hyperparameters, adjusting posterior using only obs_x and obs_y.

        Args:
            x (vector of float): locations at which to compute uncertainty
            obs_x, obs_y (vector of float): train_inputs and train_targets to use.

        Returns:
            (mean, lower, upper) values at locations x.
        """

        if self.gp_ is None:
            self.fit()

        self.set_train_data(inputs=obs_x, targets=obs_y, strict=False)
        p_mean, p_lower, p_upper = self.predict(x)

        return (p_mean, p_lower, p_upper)

    def monitoring_uncertainty(self):
        """Compute monitoring uncertainty at all inputs.

        The monitoring uncertainty of the first input is always larger than the others, because no observations have been made yet.

        Monitoring uncertainty at input location t is the width of the confidence interval when observations x[i] < t are given. This value is computed iteratively for each t=x[i], given values for x[:i].

        Args:
            resolution (float, optional): The resolution of individual curves to compute. Defaults to None.

        Returns:
            List of (mean, lower, upper) vectors of confidence limits.
        """

        # TODO: Consider for future optimization replacing FixedNoiseGaussianLiklihood with GaussianLikelihoodWithMissingObs, and then instead of re-setting the training data, set the values of train_targets to NaN incrementally, starting from the far end. This might be convertable to a matrix version, which could find all of the measurement uncertainties for a single curve in one shot, instead of one at a time, like we do here.

        x = self.train_inputs
        y = self.train_targets

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

        x = self.train_inputs
        y = self.train_targets
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
        x = self.train_inputs
        y = self.train_targets
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

        for i in range(len(self.train_inputs)):
            l, c = self.shelf_life(
                i,
                target_uncertainty=target_uncertainty,
                max_life=max_life,
                num_pts=num_pts,
            )
            shelf_life.append(l)
            complete.append(c)
        return np.asarray(shelf_life).T, np.asarray(complete).T
