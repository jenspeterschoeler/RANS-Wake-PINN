import jax.numpy as jnp
import jax    


@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred) ** 2)

def nu_t_star_func(TI, ndp):
    """Function to generate the non-dimensional eddy viscosity"""
    return (TI * ndp["z_ref"] * jnp.sqrt(3 / 2) * ndp["C_mu"] ** (-1 / 4)) / ndp["D"]

def s2u(input, min, max):
    "Scaled to unscaled variables"
    return (input / 2 + 1 / 2) * (max - min) + min

def ds2u(dx, min, max):
    "Scaled derivative to unscaled derivative"
    return (dx) * (max - min) / 2

def data_and_pinn_loss_non_jit(params, colloc, grid_data, flow_data, ndp, coords_min_max, vars_min_max, NN, data_loss_fn): 
    """An intemediary function where some of the inputs has to be supplied prior to jitting via a hidden function.

    Parameters
    ----------
    params : Flax params
        The trainable parameters of the neural network i.e. they are variables
    colloc : jnp.array
        Variable collocation points
    grid_data : jnp.array
        Variable inputs
    flow_data : jnp.array
        Variable true outputs
    ndp : dict
        Constant should be supplied in hidden function
    NN : Neural network function, this has to be supplied prior to jitting
    data_loss_fn : Similar has to be provided in hidden function prior to jitting

    Returns
    -------
    _type_
        _description_
    """
    # Unpacking constants
    Re = ndp["rho"] * 1 * ndp["D"] / ndp["mu"]
    z_min, z_max = coords_min_max['z_cyl']
    r_min, r_max = coords_min_max['r']
    TI_min, TI_max = coords_min_max['TI_amb']
    CT_min, CT_max = coords_min_max['CT']

    uz_min, uz_max = vars_min_max['U_z']
    ur_min, ur_max = vars_min_max['U_r']
    p_min, p_max = vars_min_max['P']

    uzfunc_single = lambda z, r, TI, CT: NN(params, z, r, TI, CT)[0]
    urfunc_single = lambda z, r, TI, CT: NN(params, z, r, TI, CT)[1]
    pfunc_single = lambda z, r, TI, CT: NN(params, z, r, TI, CT)[2]

    ## First order derivatives
    # Each output function is separated into its own function only capable of predicting on a single i/o pair
    #  at a time, this is to allow for the vmap to work.
    uz_z_single = lambda z, r, TI, CT: jax.grad(uzfunc_single, argnums=0)(z, r, TI, CT)
    uz_r_single = lambda z, r, TI, CT: jax.grad(uzfunc_single, argnums=1)(z, r, TI, CT)
    ur_z_single = lambda z, r, TI, CT: jax.grad(urfunc_single, argnums=0)(z, r, TI, CT)
    ur_r_single = lambda z, r, TI, CT: jax.grad(urfunc_single, argnums=1)(z, r, TI, CT)

    # Pressure is not used in the second order derivatives, hence the single i/o pair function is not saved
    p_z_single = lambda z, r, TI, CT: jax.grad(pfunc_single, argnums=0)(z, r, TI, CT)
    p_r_single = lambda z, r, TI, CT: jax.grad(pfunc_single, argnums=1)(z, r, TI, CT)

    ## Second order derivatives
    uz_zz_single = lambda z, r, TI, CT: jax.grad(uz_z_single, argnums=0)(z, r, TI, CT)
    uz_rr_single = lambda z, r, TI, CT: jax.grad(uz_r_single, argnums=1)(z, r, TI, CT)
    ur_zz_single = lambda z, r, TI, CT: jax.grad(ur_z_single, argnums=0)(z, r, TI, CT)
    ur_rr_single = lambda z, r, TI, CT: jax.grad(ur_r_single, argnums=1)(z, r, TI, CT)

    # Unscaled Functions for Physics
    r_cus = lambda r: s2u(r, r_min, r_max)
    uz_cus = lambda z, r, TI, CT: s2u(uzfunc_single(z, r, TI, CT), uz_min, uz_max)
    ur_cus = lambda z, r, TI, CT: s2u(urfunc_single(z, r, TI, CT), ur_min, ur_max)

    uz_z_cus = lambda z, r, TI, CT: ds2u(uz_z_single(z, r, TI, CT), uz_min, uz_max)
    uz_r_cus = lambda z, r, TI, CT: ds2u(uz_r_single(z, r, TI, CT), uz_min, uz_max)
    ur_z_cus = lambda z, r, TI, CT: ds2u(ur_z_single(z, r, TI, CT), ur_min, ur_max)
    ur_r_cus = lambda z, r, TI, CT: ds2u(ur_r_single(z, r, TI, CT), ur_min, ur_max)
    p_z_cus = lambda z, r, TI, CT: ds2u(p_z_single(z, r, TI, CT), p_min, p_max)
    p_r_cus = lambda z, r, TI, CT: ds2u(p_r_single(z, r, TI, CT), p_min, p_max)

    uz_zz_cus = lambda z, r, TI, CT: ds2u(uz_zz_single(z, r, TI, CT), uz_min, uz_max)
    uz_rr_cus = lambda z, r, TI, CT: ds2u(uz_rr_single(z, r, TI, CT), uz_min, uz_max)
    ur_zz_cus = lambda z, r, TI, CT: ds2u(ur_zz_single(z, r, TI, CT), ur_min, ur_max)
    ur_rr_cus = lambda z, r, TI, CT: ds2u(ur_rr_single(z, r, TI, CT), ur_min, ur_max)


    def eval_physics_se_single_io_pair(z, r, TI, CT):
        """This function evaluates the squared error of the physics information for a single i/o pair.
        The mean is outside this function in order of allowing vmap on this function
        input: z, r, TI, CT (unscaled)
        output: se_pinn
        """
        uz = uz_cus(z, r, TI, CT)
        ur = ur_cus(z, r, TI, CT)

        p_r = p_r_cus(z, r, TI, CT)
        p_z = p_z_cus(z, r, TI, CT)

        uz_z = uz_z_cus(z, r, TI, CT)
        uz_r = uz_r_cus(z, r, TI, CT)
        ur_r = ur_r_cus(z, r, TI, CT)
        ur_z = ur_z_cus(z, r, TI, CT) 

        uz_zz = uz_zz_cus(z, r, TI, CT)
        uz_rr = uz_rr_cus(z, r, TI, CT)

        ur_rr = ur_rr_cus(z, r, TI, CT)
        ur_zz = ur_zz_cus(z, r, TI, CT)

        continuity = uz_z + ur_r / r
        r_momentum = ur*ur_r + uz*ur_z + p_r - (1/Re + nu_t_star_func(TI, ndp)) * (ur_rr + ur_r / r + ur_zz - ur / r**2)
        z_momentum = ur*uz_r + uz*uz_z + p_z - (1/Re + nu_t_star_func(TI, ndp)) * (uz_rr + uz_r / r + uz_zz)

        se_pinn = continuity**2 + r_momentum**2 + z_momentum**2
        return se_pinn
    
    # Obtaining vmapped and jitted physics function
    eval_physics_se = jax.vmap(eval_physics_se_single_io_pair, in_axes=(0, 0, 0, 0)) # "se" is square error
    mse_pinn_fn = lambda z_c, r_c, TI_c, CT_c: jnp.mean(eval_physics_se(z_c, r_c, TI_c, CT_c), axis=0)

    mse_data = data_loss_fn(params, grid_data, flow_data)
    z_c, r_c, TI_c, CT_c = colloc[:, 0], colloc[:, 1], colloc[:, 2], colloc[:, 3]
    z_cu = s2u(z_c, z_min, z_max)
    r_cu = s2u(r_c, r_min, r_max)
    TI_cu = s2u(TI_c, TI_min, TI_max)
    CT_cu = s2u(CT_c, CT_min, CT_max)
    mse_pinn = mse_pinn_fn(z_cu, r_cu, TI_cu, CT_cu)
    return mse_data, mse_pinn
