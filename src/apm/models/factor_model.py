from flax import nnx
import jax


class FactorModel(nnx.Module):
    def __init__(
        self, d_factors: int, d_assets: int, use_bias: bool, rngs: nnx.Rngs
    ):
        self.linear = nnx.Linear(
            d_factors, d_assets, rngs=rngs, use_bias=use_bias
        )

    def __call__(self, x: jax.Array):
        return self.linear(x)
