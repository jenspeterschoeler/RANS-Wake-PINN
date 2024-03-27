import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
# import orbax
import orbax.checkpoint
from omegaconf import DictConfig

from typing import Sequence
import re

import logging
logger = logging.getLogger(__name__)



class MLP(nn.Module):
    """Simple MLP with tanh activation function
       Notice this uses expclit declaration of variables could also use nn.compact as well, 
       see https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html#reasons-to-prefer-using-nn-compact """
    
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(features=feat, use_bias=True, kernel_init=jax.nn.initializers.glorot_normal()) for feat in self.features]
    
    def __call__(self, inputs):
        x = inputs
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers)-1:
                x = jnp.tanh(x)
        return x

def setup_variable_MLP(cfg, in_layers=None, out_layers=None):
    """Setup an MLP with varying layer size

    Parameters
    ----------
    cfg : structured dict with hyper parameters for model

    Returns
    -------
    Flax ANN model
    """
    keys = cfg.model.keys()
    nkeys = []
    nvals = []

    for key in keys:
        if (re.search('node', key) and cfg.model[key] > 0):
            nkeys.append(key)
            nvals.append(cfg.model[key])
    logger.debug(nvals)
    if in_layers != None:
        layers_list = [in_layers]
    else:
        layers_list = []

    hidden_layers = nvals
    layers_list = layers_list + hidden_layers
    if out_layers != None:
        layers_list = layers_list + [out_layers]

    model = MLP(layers_list)
    return model

def setup_uniform_MLP(cfg, in_layers=None, out_layers=None):
    """Setup an MLP with uniform number of nodes in each layer
    """
    n_layers = cfg.model.n_layers
    n_nodes = cfg.model.n_nodes
    if in_layers != None:
        layers_list = [in_layers]
    else:
        layers_list = []

    for _ in range(n_layers):
        layers_list.append(n_nodes)
    
    if out_layers != None: 
        layers_list.append(out_layers)
    model = MLP(layers_list)
    return model

def setup_MLP(cfg, in_layers=None, out_layers=None):
    """Setup an MLP model
    Parameters
    ----------
    cfg : OmegaConf
        configuration dict like object
    in_layers : _type_, optional
        input layer size, by default None
    out_layers : _type_, optional
        output layer size, by default None

    Returns
    -------
        Flax MLP model
    """
    if cfg.model.type == 'MLP_uniform_layersize':
        model = setup_uniform_MLP(cfg, in_layers=in_layers, out_layers=out_layers)
    elif cfg.model.type == 'MLP_variable_layersize':
        model = setup_variable_MLP(cfg, in_layers=in_layers, out_layers=out_layers)
    else:
        raise NotImplementedError('Model type not implemented')
    return model


def load_model(path):
    """Load a model from a checkpoint
    Parameters
    ----------
    path : str
        path to checkpoint

    Returns
    -------
    Flax MLP model, model parameters, config dict
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(path)
    cfg = DictConfig(raw_restored['config'])
    if cfg.model.type == 'MLP_uniform_layersize' or cfg.model.type == 'MLP_variable_layersize':
        model = setup_MLP(cfg, 
                          in_layers=len(cfg.data.input_coords), 
                          out_layers=len(cfg.data.output_vars))
    else:
        raise NotImplementedError('Model type not implemented')
    params = raw_restored['model']['params']
    return model, params, cfg


class SoftAdapt:
    """Adaptive Loss Weigthing with SoftAdapt algorithm.
    Currently only supports a single historical loss iteration.
    based on https://arxiv.org/abs/1912.12355"""
    def __init__(self, beta=0.1, epsilon=1e-8, loss_weighted=False):
        self.beta = beta
        self.epsilon = epsilon
        self.loss_weighted=loss_weighted

    def compute_alphas(self, rates_of_change, loss_components):
        """Compute the loss weights
        Parameters
        ----------
        rates_of_change : jnp array
            loss rates of change of current iteration for each loss component (j)

        loss_compoents : jnp array
            loss components of current iteration

        Returns
        -------
        alpha_weights : jnp array
            loss weights for current iteration
        """

        # Shift by max value to avoid numerical issues
        max_rate_of_change = jnp.max(rates_of_change)
        rates_of_change = rates_of_change - max_rate_of_change
        
        denominator = jnp.sum(jnp.exp(self.beta*rates_of_change)) + self.epsilon
        alpha_weights = jnp.exp(self.beta*rates_of_change) / denominator

        if self.loss_weighted:
            alpha_weights = (alpha_weights * loss_components) / (jnp.sum(alpha_weights * loss_components) + self.epsilon)

        return alpha_weights
    

class BackwardsDifference:
    """Finite Difference calculator for derivatives in backward direction"""
    def __init__(self, order=5):
        self.order = order
        if self.order == 5:
            # in order [-5, -4, -3, -2, -1, 0]
            self.bd_coeffs = jnp.array([-1/5, 5/4, -10/3, 5, -5, 137/60])
        else:
            raise NotImplementedError('Only order 5 is implemented')

    def compute_bd(self, x):
        """Compute the derivative of x
        Parameters
        ----------
        x : jnp array
            history of x in order i = [-order, ..., -3, -2, -1, 0]

        Returns
        -------
        jnp array
            derivative of x
        """
        return jnp.dot(self.bd_coeffs, x)

def setup_SoftAdapt_and_BD(cfg):
    """Setup SoftAdapt and BackwardsDifference objects from config
    Parameters
    ----------
    cfg : OmegaConf
        configuration dict like object

    Returns
    -------
    SoftAdapt, BackwardsDifference
    """
    if cfg.optimizer.loss_balancing.type == 'softadapt':
        beta = cfg.optimizer.loss_balancing.params.beta
        epsilon = cfg.optimizer.loss_balancing.params.epsilon
        loss_weighted = cfg.optimizer.loss_balancing.params.loss_weighted
        loss_balancer = SoftAdapt(beta=beta, epsilon=epsilon, loss_weighted=loss_weighted)
        bd_order = cfg.optimizer.loss_balancing.params.bd_order
        bd = BackwardsDifference(order=bd_order)
    return loss_balancer, bd

if __name__ == "__main__":
    model_path = "../Experimental/14-13-25/final_model"
    model, params, cfg = load_model(model_path)