import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple

def init_mlp_params(layer_sizes: List[int], key: jax.Array) -> Dict[str, List[jax.Array]]:
    keys = random.split(key, len(layer_sizes) - 1)
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
        stddev = jnp.sqrt(2.0 / (in_dim + out_dim))
        weights.append(random.normal(keys[i], (in_dim, out_dim)) * stddev)
        biases.append(jnp.zeros((out_dim,)))
    return {"weights": weights, "biases": biases}

@jax.jit
def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))

@jax.jit
def layer_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)

def mlp_forward(params: Dict, X: jnp.ndarray, key: jax.Array = None, 
                deterministic: bool = True, dropout_rate: float = 0.025) -> jnp.ndarray:
    weights, biases = params["weights"], params["biases"]
    num_layers = len(weights)
    dropout_key = key
    
    for i in range(num_layers - 1):
        X = jnp.dot(X, weights[i]) + biases[i]
        if i < 2:
            X = layer_norm(X)
        X = mish(X)
        if i == 1 and not deterministic and dropout_key is not None:
            dropout_key, subkey = random.split(dropout_key)
            mask = random.bernoulli(subkey, 1.0 - dropout_rate, X.shape)
            X = (X * mask) / (1.0 - dropout_rate)
            
    return jnp.dot(X, weights[-1]) + biases[-1]

@jax.jit
def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    error = y_true - y_pred
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return jnp.mean(0.5 * jnp.square(quadratic) + delta * linear)

@jax.jit
def loss_fn(params: Dict, X: jnp.ndarray, y: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    y_pred = mlp_forward(params, X, key=key, deterministic=False)
    return huber_loss(y, y_pred)

@jax.jit
def update_step(params: Dict, X: jnp.ndarray, y: jnp.ndarray, key: jax.Array, lr: float) -> Tuple[Dict, jnp.ndarray]:
    loss_val, grads = jax.value_and_grad(loss_fn)(params, X, y, key)
    updated_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return updated_params, loss_val

def get_batches(X, Y, batch_size, key):
    num_samples = X.shape[0]
    steps_per_epoch = num_samples // batch_size
    perms = random.permutation(key, num_samples)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    return [(X[p], Y[p]) for p in perms]
