import numpy as np

from gsm.config import FitConfig, GaussianMixtureSetting
from gsm.data import Dataset
from gsm.variational import fit_variational, sample_gaussian_mixture_posterior


def test_gaussian_mixture_variational_smoke():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GaussianMixtureSetting(
        n_components=2,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=50,
        learning_rate=0.03,
        n_restarts=1,
        tol=0.0,
        n_elbo_samples=4,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.elbo_history[-1] > result.elbo_history[0]
    assert result.posterior is not None
    assert np.isfinite(result.posterior.log_std.mean_coef).all()
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(y)))
    assert result.predictive_mean is not None
    assert result.predictive_variance is not None


def test_gaussian_mixture_variational_with_ard():
    y = np.asarray([-2.1, -1.9, -2.0, 1.8, 2.0, 2.2])[:, None]
    x = np.linspace(-1.0, 1.0, len(y))
    X = np.column_stack([np.ones(len(y)), x])
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const", "x"))
    setting = GaussianMixtureSetting(
        n_components=2,
        covs=((0, 1), (0, 1)),
        covs_mix=(0, 1),
        standardize=0,
    )
    fit = FitConfig(
        seed=123,
        max_iter=30,
        learning_rate=0.02,
        n_restarts=1,
        tol=0.0,
        use_ard=True,
        n_elbo_samples=4,
    )

    result = fit_variational(dataset, setting, fit)

    assert np.isfinite(result.elbo_history).all()
    assert result.responsibilities is not None
    assert result.responsibilities.shape == (len(y), 2)
    np.testing.assert_allclose(result.responsibilities.sum(axis=1), np.ones(len(y)))


def test_gaussian_mixture_posterior_sampling_shapes():
    y = np.asarray([-1.0, 0.0, 1.0])[:, None]
    X = np.ones((len(y), 1))
    dataset = Dataset(y=y, X=X, y_name="y", x_names=("Const",))
    setting = GaussianMixtureSetting(
        n_components=1,
        covs=((0,), (0,)),
        covs_mix=(0,),
        standardize=0,
    )
    fit = FitConfig(seed=123, max_iter=5, learning_rate=0.01, n_elbo_samples=2)

    result = fit_variational(dataset, setting, fit)

    assert result.posterior is not None
    samples = sample_gaussian_mixture_posterior(result.posterior, seed=123, n_samples=7)
    assert samples["mean_coef"].shape == (7, 1, 1)
    assert samples["log_variance_coef"].shape == (7, 1, 1)
    assert samples["gating_coef"].shape == (7, 0, 1)


def test_predictive_variance_includes_parameter_uncertainty(monkeypatch):
    import jax.numpy as jnp
    import gsm.vi.engine as engine
    from gsm.models.gaussian import GaussianMixtureParams
    from gsm.vi.common import VariationalResult
    from gsm.vi.gaussian import GaussianMixturePosterior, fit_gaussian_mixture_vb

    params = GaussianMixtureParams(jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((0, 1)))
    log_std = GaussianMixtureParams(jnp.full((1, 1), jnp.log(2.)),
                                   jnp.full((1, 1), -30.), jnp.zeros((0, 1)))
    posterior = GaussianMixturePosterior(params, log_std)
    monkeypatch.setattr(engine, "optimize_restarts", lambda *a: VariationalResult(
        params=params, posterior=posterior, elbo_history=np.array([0.]), converged=True,
    ))
    data = Dataset(np.array([[-1.], [0.], [1.]]), np.ones((3, 1)), "y", ("Const",))
    result = fit_gaussian_mixture_vb(
        data, GaussianMixtureSetting(n_components=1),
        FitConfig(n_predictive_samples=4096),
    )
    np.testing.assert_allclose(result.predictive_mean, np.zeros(3), atol=.1)
    np.testing.assert_allclose(result.predictive_variance, np.full(3, 5.), atol=.2)
    assert result.n_predictive_samples == 4096
