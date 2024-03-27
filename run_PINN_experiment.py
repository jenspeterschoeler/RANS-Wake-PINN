import hydra
import logging

import os
from utils import jax_flax as jf_utils
from utils.data import get_data_manager
from utils.training import MSE, data_and_pinn_loss_non_jit
from omegaconf import DictConfig, OmegaConf
from pandas import json_normalize
import jax.numpy as jnp
import jax
from jax.tree_util import tree_map
from torch.utils import data
import optax
import orbax.checkpoint as ocp
import numpy as np
from tqdm import tqdm
from flax.training import orbax_utils
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from utils.plotting import Plotter
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)

## Test configurations
config_name = "PINN_local_v0"           # Test config
# config_name = "PINN_submitit_exp"     # Test config

## Data-driven configurations using full dataset (Data-driven grid search)
# config_name = "PINN_exp_v0"    # Data-driven (network size)
# config_name = "PINN_pre_exp2"  # Data-driven (Batch size and learning rate)

## PINN experiments
# config_name = "PINN_exp1"      # First true experiment
# config_name = "PINN_exp1_data" # First exp but with only truncated data
# config_name = "PINN_exp2"      # Second true experiment (refactored slightly for better implementation of batch size)



config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "configurations"))
data_path = './Data/RANS_1wt_irot_v2.nc'


@hydra.main(config_path=config_path, config_name=config_name, version_base="1.3")
def main(cfg: DictConfig):
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # ================ Logging ================
    log.info(f"Configuration content:\n{OmegaConf.to_yaml(cfg)}")
    hydra_name = hydra_output_dir.split('/')[-2:]
    hydra_name = '_'.join(hydra_name)
    writer = SummaryWriter(log_dir=f"{hydra_output_dir}/tensorboard")

    # ================ Data ================
    DM = get_data_manager(cfg)
    # Re is constanc beacuse U_ref is constant (which is okay due to Reynolds similarity i.e. 1/nu_t >> 1/Re)
    n_in = len(cfg.data.input_coords)
    n_out = len(cfg.data.output_vars)
    ndp = DM.non_dim_vals
    coords_min_max = DM.coords_min_max
    vars_min_max = DM.vars_min_max


    train_data_loader = DM.get_train_dataloader(batch_size=cfg.optimizer.batch,
                                                physics_on=cfg.optimizer.physics.toggled,
                                                data_ratio=cfg.optimizer.data_ratio,
                                                colloc_data_ratio=cfg.optimizer.physics.colloc_data_ratio)
    X_val, y_val = DM.get_validation_set()
    test_df_list = DM.get_test_set()

    # ================ Model ================
    NN_model = jf_utils.setup_MLP(cfg, in_layers=n_in, out_layers=n_out)

    key1, key2 = jax.random.split(jax.random.PRNGKey(420), num=2)
    dummy_in = jax.random.normal(key1, (n_in,))
    params = NN_model.init(key2, dummy_in)
    optimizer = optax.adam(cfg.optimizer.lr)

    # ============ Loss functions ============

    @jax.jit
    def NN(params, z, r, TI, CT):
        """Evaluation of the neural network"""
        inp = jnp.array([z, r, TI, CT])
        func = NN_model.apply(params, inp)
        return func

    @jax.jit
    def data_loss_fn(params, X, y):
        """Setup MSE data loss function"""
        pred = NN_model.apply(params, X)
        return MSE(y, pred)
    
    @jax.jit
    def data_and_pinn_loss(params, colloc, grid_data, flow_data):
        """This function is the jitted version of the data and pinn losses.
        It is required to be jitted inside the main function to include the specific 
        neural network/data configuration Hydra is currently running."""
        return data_and_pinn_loss_non_jit(params, colloc, grid_data, flow_data, ndp, coords_min_max, vars_min_max, NN, data_loss_fn)

    @jax.jit
    def loss_fun(params, colloc_outer, grid_data, flow_data, alphas):
        """Combined loss fucntion with weights."""
        data_loss, pinn_loss = data_and_pinn_loss(params, colloc_outer, grid_data, flow_data)
        weighted_loss = alphas[0]*data_loss + alphas[1]*pinn_loss
        return weighted_loss, (data_loss, pinn_loss)

    @jax.jit
    def update(opt_state, colloc_outer, grid_data, flow_data, alphas):
        """Update function for the optimization loop."""
        value_and_grad_fn = jax.value_and_grad(loss_fun, 0, has_aux=True)
        (w_loss, (data_loss, pinn_loss)), grads = value_and_grad_fn(opt_state.params, colloc_outer, grid_data, flow_data, alphas)
        opt_state = opt_state.apply_gradients(grads=grads)
        return opt_state, data_loss, pinn_loss

    @jax.jit
    def update_data(opt_state, grid_data, flow_data):
        """Update function for the optimization loop, only considering data loss.
        Requires flag in config 'cfg.optimizer.physics.toggled=False'."""
        data_loss, grads = jax.jit(jax.value_and_grad(data_loss_fn, 0))(opt_state.params, grid_data, flow_data)
        opt_state = opt_state.apply_gradients(grads=grads)
        return opt_state, data_loss

    def train_epoch(opt_state, training_data_loader, val_data, alphas, ignore_pinn=False):
        """Train for a single epoch."""
        X_val, y_val = val_data
        data_losses = []
        pinn_losses = []
        weighted_data_losses = []
        weighted_pinn_losses = []
        if ignore_pinn:
            for X_batch, y_batch, _ in training_data_loader:
                opt_state, data_loss_batch = update_data(opt_state, X_batch, y_batch)
                data_losses.append(data_loss_batch)
                pinn_loss_batch = np.nan
                pinn_losses.append(pinn_loss_batch)
                weighted_data_losses.append(alphas[0]*data_loss_batch)
                weighted_pinn_losses.append(alphas[1]*pinn_loss_batch)
            
            pinn_loss = np.nan
            weighted_data_loss = np.nan
            weighted_pinn_loss = np.nan
        else:
            for X_batch, y_batch, X_c_batch in training_data_loader:
                opt_state, data_loss_batch, pinn_loss_batch = update(opt_state, X_c_batch, X_batch, y_batch, alphas)
                data_losses.append(data_loss_batch)
                pinn_losses.append(pinn_loss_batch)
                weighted_data_losses.append(alphas[0]*data_loss_batch)
                weighted_pinn_losses.append(alphas[1]*pinn_loss_batch)

            pinn_loss = np.mean(pinn_losses)
            weighted_data_loss = np.mean(weighted_data_losses)
            weighted_pinn_loss = np.mean(weighted_pinn_losses)
        data_loss = np.mean(data_losses)
        val_loss = data_loss_fn(opt_state.params, X_val, y_val)
        return opt_state, data_loss, pinn_loss, val_loss, weighted_data_loss, weighted_pinn_loss

    def train_epoch_data(opt_state, training_data_loader, val_data):
        """Train for a single epoch, only considering data loss.
        Requires flag in config 'cfg.optimizer.physics.toggled=False'."""
        X_val, y_val = val_data
        data_losses = []

        for X_batch, y_batch in training_data_loader:
            opt_state, data_loss_batch = update_data(opt_state, X_batch, y_batch)
            data_losses.append(data_loss_batch)
        data_loss = np.mean(data_losses)
        val_loss = data_loss_fn(opt_state.params, X_val, y_val)

        return opt_state, data_loss, val_loss

    # Setting up checkpoint manager using the Orbax checkpoint framework 
    #   (Google JAX utility library - flax suggested)
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_dir = f"{hydra_output_dir}/checkpoints"
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    init_save_args = True
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, orbax_checkpointer, options)

    #  Setting up early stopping
    early_stop = EarlyStopping(min_delta=cfg.optimizer.early_stop.criteria, 
                               patience=cfg.optimizer.early_stop.patience)
    
    # Initialize optimizer with first train state
    opt_state = TrainState.create(apply_fn=loss_fun, params=params, tx=optimizer)

    # Intialize loss history 
    data_epoch_losses = []
    val_epoch_losses = []
    val_loss_hist = 99999 # Arbitrary high value (normal range ~1e-3 to 1e-6)
    if cfg.optimizer.physics.toggled:
        # Initilization only run when physics is toggled
        pinn_epoch_losses = []
        weighted_data_epoch_losses = []
        weighted_pinn_epoch_losses = []
        if cfg.optimizer.loss_balancing.type == 'softadapt': # Setup for softadapt loss balancing
            loss_balancer, bd = jf_utils.setup_SoftAdapt_and_BD(cfg)

            @jax.jit
            def compute_bd(loss_hist):
                """Jitted backward difference for weight loss balancing"""
                return bd.compute_bd(loss_hist)

            @jax.jit
            def compute_alphas(rates_of_change, loss_components):
                """Jitted computation of loss weights (alphas)"""
                return loss_balancer.compute_alphas(rates_of_change, loss_components)

            alphas = jnp.ones(2)
            old_rates_of_change = None
        elif cfg.optimizer.loss_balancing.type == 'fixed': # Setup for constant loss weights i.e. fixed weights
            alphas = jnp.array([cfg.optimizer.loss_balancing.params.alpha_data, cfg.optimizer.loss_balancing.params.alpha_physics])

    # ================ Training loop ================
    for epoch in tqdm(range(cfg.optimizer.epochs)):
        train_data_loader.shuffle_data()

        if cfg.optimizer.physics.toggled:
            if epoch < cfg.optimizer.physics.delayed_start:
                opt_state, data_loss, pinn_loss, val_loss, weighted_data_loss, weighted_pinn_loss = train_epoch(opt_state, train_data_loader, (X_val, y_val), alphas, ignore_pinn=True)
            else:
                opt_state, data_loss, pinn_loss, val_loss, weighted_data_loss, weighted_pinn_loss = train_epoch(opt_state, train_data_loader, (X_val, y_val), alphas, ignore_pinn=False)
            pinn_epoch_losses.append(pinn_loss)
            weighted_data_epoch_losses.append(weighted_data_loss)
            weighted_pinn_epoch_losses.append(weighted_pinn_loss)

            print_message = f"Epoch = {epoch},\t w_loss = {(weighted_data_loss+weighted_pinn_loss):.3e},\t data loss = {data_loss:.3e},\t pinn loss = {pinn_loss:.3e},\t val loss = {val_loss:.3e}"
        else:
            opt_state, data_loss, val_loss = train_epoch_data(opt_state, train_data_loader, (X_val, y_val))
            print_message = f"Epoch = {epoch},\t data loss = {data_loss:.3e},\t val loss = {val_loss:.3e}"

        data_epoch_losses.append(data_loss)
        val_epoch_losses.append(val_loss)
        
        if epoch % 1 == 0: # Prints to console every 1 epoch (can be adjusted to any number)
            print(print_message) # TODO - Use logging instead of print? debug only?

        early_stop = early_stop.update(val_loss)

        if val_loss <= val_loss_hist: # Save checkpoint if validation loss is lower than previous lowest
            ckpt = {"model": opt_state,
                    "config": OmegaConf.to_container(cfg),
                    "data": data_path,}
            if init_save_args: # First time run initialization of save_args
                save_args = orbax_utils.save_args_from_target(ckpt)
                init_save_args = False # Prevents reinitialization
            checkpoint_manager.save(epoch, ckpt, save_kwargs={"save_args": save_args})
            val_loss_hist = val_loss
        

        writer.add_scalar("Loss/data", data_loss, epoch)
        writer.add_scalar("Loss/val", np.asarray(val_loss), epoch)

        if cfg.optimizer.physics.toggled:
            if epoch >= cfg.optimizer.physics.delayed_start:
                p_epoch = epoch - cfg.optimizer.physics.delayed_start
                if cfg.optimizer.loss_balancing.type == 'softadapt':
                    if p_epoch >= bd.order:
                        data_loss_hist = jnp.array(data_epoch_losses[-(bd.order+1):])
                        pinn_loss_hist = jnp.array(pinn_epoch_losses[-(bd.order+1):])
                        rates_of_change = []
                        for loss_hist in (data_loss_hist, pinn_loss_hist):
                            rates_of_change.append(compute_bd(loss_hist))
                        rates_of_change = jnp.array(rates_of_change)
                        losses = jnp.array([data_loss, pinn_loss])
                        if old_rates_of_change is not None:
                            alphas = compute_alphas(rates_of_change, losses)
                        old_rates_of_change = rates_of_change

            writer.add_scalar("Loss/alpha_data", np.asarray(alphas[0]), epoch)
            writer.add_scalar("Loss/alpha_pinn", np.asarray(alphas[1]), epoch)
            writer.add_scalar("Loss/pinn", pinn_loss, epoch)
            writer.add_scalar("Loss/tot", data_loss + pinn_loss, epoch)
            writer.add_scalar("Loss/weighted_tot", weighted_data_loss + weighted_pinn_loss, epoch)
            writer.add_scalar("Loss/weighted_data", weighted_data_loss, epoch)
            writer.add_scalar("Loss/weighted_pinn", weighted_pinn_loss, epoch)

        

        if early_stop.should_stop:
            log.info(f"Met early stopping criteria, breaking...,\n epoch = {epoch}, \t val loss = {val_loss:.3e}, \t delta_stop = {early_stop.delta:.3e}")
            break

    # ================ Save ================
    metric_dict = {"Loss/data": data_loss, 
                   "Loss/val": float(np.asarray(val_loss)),
                   }
    if cfg.optimizer.physics.toggled:
        physics_metric_dict = {"Loss/pinn": pinn_loss,
                               "Loss/tot": data_loss + pinn_loss,
                               "Loss/weighted_tot": weighted_data_loss + weighted_pinn_loss,
                               "Loss/weighted_data": weighted_data_loss,
                               "Loss/weighted_pinn": weighted_pinn_loss,
                               "Loss/alpha_data": float(np.asarray(alphas[0])),
                               "Loss/alpha_pinn": float(np.asarray(alphas[1])),}
        metric_dict.update(physics_metric_dict)
        
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_conf_dict = json_normalize(conf_dict, sep="/").to_dict("index")[0]
    for key, val in flat_conf_dict.items(): # Flattens lists
        if isinstance(val, list):
            flat_conf_dict[key] = "list = "+str(val)


    final_model_ckpt = {
        "model": opt_state,
        "config": OmegaConf.to_container(cfg),
        "data": data_path,
    }

    log.info(f"Attempting to save final model to: {hydra_output_dir}/final_model")
    final_save_args = orbax_utils.save_args_from_target(final_model_ckpt)
    orbax_checkpointer.save(f"{hydra_output_dir}/final_model", final_model_ckpt, save_args=final_save_args)
    # ==================== Test ====================
    log.info(f"Attempting to process final model in: {hydra_output_dir}/final_model")
    test_losses_flowcases = []
    for flowcase in range(len(test_df_list)):
        test_df = test_df_list[flowcase].copy()
        # setup input data
        z_test = test_df['z_cyl'].to_numpy()
        r_test = test_df['r'].to_numpy()
        TI_test = test_df['TI_amb'].to_numpy()
        CT_test = test_df['CT'].to_numpy()

        # setup output data
        uz_test = test_df['U_z'].to_numpy()
        ur_test = test_df['U_r'].to_numpy()
        p_test = test_df['P'].to_numpy()

        X_test = jnp.concatenate((z_test.reshape(-1, 1), r_test.reshape(-1, 1), TI_test.reshape(-1, 1), CT_test.reshape(-1, 1)), axis=1)
        y_test = jnp.concatenate((uz_test.reshape(-1, 1), ur_test.reshape(-1, 1), p_test.reshape(-1, 1)), axis=1)

        test_loss = data_loss_fn(opt_state.params, X_test, y_test)
        test_losses_flowcases.append(test_loss)
        log.info(f"Test loss flowcase {flowcase} = {test_loss:.3e}")
        metric_dict[f"Loss/test_flowcase_{flowcase}"] = float(np.asarray(test_loss))
        # writer.add_hparams(flat_conf_dict, {f"Loss/test_flowcase_{flowcase}": float(np.asarray(test_loss))})
    log.info(f"Test loss = {np.mean(test_losses_flowcases):.3e}")
    metric_dict[r"Loss/test_tot"] = float(np.mean(np.asarray(test_losses_flowcases)))
    
    writer.add_hparams(flat_conf_dict, metric_dict, run_name=hydra_name)

    # ================ Plotting ================
    log.info(f"Plotting test cases with final model in: {hydra_output_dir}/final_model")
    plotter = Plotter(DM, model_path=f"{hydra_output_dir}/final_model")
    losses_dict = {"Data": data_epoch_losses, "Val": val_epoch_losses}
    if cfg.optimizer.physics.toggled:
        losses_dict["PINN"] = pinn_epoch_losses
        losses_dict["Total"] = np.array(data_epoch_losses)+np.array(pinn_epoch_losses)
        losses_dict["Weighted Data"] = weighted_data_epoch_losses
        losses_dict["Weighted_PINN"] = weighted_pinn_epoch_losses
        losses_dict["Weighted Total"] = np.array(weighted_data_epoch_losses)+np.array(weighted_pinn_epoch_losses)

    fig, ax = plt.subplots()
    for label, loss in losses_dict.items():
        ax.semilogy(loss, label=label)
    ax.legend()
    ax.set_xlabel('Epoch [-]')
    ax.set_ylabel(r'$\mathcal{L}$')
    plt.savefig(f"{hydra_output_dir}/final_model/loss_hist.png")

    for flowcase in range(len(test_df_list)):
        for var in DM.vars:
            fig, axes = plotter.plot_pred_triplet(var, flowcase=flowcase)
            plt.suptitle(f'Flowcase: {flowcase}')
            plt.savefig(f"{hydra_output_dir}/final_model/contour_{flowcase}_{var}.png")
            plt.close(fig)

    log.info("Main script/Main function complete!")

if __name__ == "__main__":
    main()

    if 0:  
        # Load and compose hydra cfg in ipython for running in interactive mode
        from hydra import initialize, initialize_config_module, initialize_config_dir, compose
        config_path = os.path.relpath(os.path.join(os.path.dirname(__file__), "configurations"))
        with initialize(version_base="1.3", config_path=config_path):
            cfg = compose(config_name=config_name)
            print(cfg)
