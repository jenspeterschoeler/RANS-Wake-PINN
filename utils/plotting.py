from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./'))
from utils.jax_flax import load_model
from utils.data import TensorBoardLoader
from utils.training import data_and_pinn_loss_non_jit
import matplotlib.style
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.style.use('classic')
mpl.rc('image', cmap='viridis')
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format
class Plotter:
    """Plotter class for plotting flow contours and other things"""
    def __init__(self, DM, model_path=None, get_complete_set=False, test_special_flowcase=None, include_MSE_pinn=True):
        self.DM = DM
        self.model_path = model_path
        if "final_model" in self.model_path:
            dirs = self.model_path.split("/")[:-1]
            dirs.append("tensorboard")
            self.tensorboard_path = "/".join(dirs)
        elif "checkpoints" in self.model_path:
            dirs = self.model_path.split("/")[:-3]
            dirs.append("tensorboard")
            self.tensorboard_path = "/".join(dirs)
        self.grid_shape = (len(self.DM.ds.z_cyl), len(self.DM.ds.r))
        testdf = self.DM.get_test_set()[0]
        testdf_unscaled = self.DM.unscale_inputs(testdf[self.DM.coords])
        self.z_grid = testdf_unscaled['z_cyl'].to_numpy().reshape(self.grid_shape)
        self.r_grid = testdf_unscaled['r'].to_numpy().reshape(self.grid_shape)
        self.exclude_mask = testdf['exclusion_mask'].to_numpy().reshape(self.grid_shape)

        if test_special_flowcase is not None:
            complete_set = self.DM.get_complete_set()
            self.test_df_list = [complete_set[i] for i in test_special_flowcase]
        else:
            self.test_df_list = self.DM.get_test_set()

        if get_complete_set:
            self.df_all_scaled_list = self.DM.get_complete_set()

        if self.model_path is not None:
            model, params, cfg = load_model(self.model_path)
            self.model = model
            self.params = params
            self.cfg = cfg
            self.predictions_list = self._predict_test_set()
            if get_complete_set:
                self.df_all_predictions_list = self._predict_complete_set()
        self.unscaled_test_set = self._unscale_test_set()
        if get_complete_set:
            self.unscaled_df_all = self._unscale_complete_set()
        self.TB_loader = TensorBoardLoader(self.tensorboard_path)
        if include_MSE_pinn:
            self.MSE_pinn = self._predict_MSE_pinn()
        else:
            self.MSE_pinn = np.nan

    def _unscale_test_set(self):
        unscaled_test_df_list = []
        for test_df in self.test_df_list:
            unscaled_test_df_coords = self.DM.unscale_inputs(test_df[self.DM.coords])
            unscaled_test_df_vars = self.DM.unscale_outputs(test_df[self.DM.vars])
            unscaled_test_df = pd.concat([unscaled_test_df_coords, unscaled_test_df_vars], axis=1)
            unscaled_test_df_list.append(unscaled_test_df)
        return unscaled_test_df_list

    def _unscale_complete_set(self):
        unscaled_df_list = []
        for df in self.df_all_scaled_list:
            unscaled_df_coords = self.DM.unscale_inputs(df[self.DM.coords])
            unscaled_df_vars = self.DM.unscale_outputs(df[self.DM.vars])
            unscaled_df = pd.concat([unscaled_df_coords, unscaled_df_vars], axis=1)
            unscaled_df_list.append(unscaled_df)
        return unscaled_df_list

    def _predict_test_set(self):
        pred_df_list = []
        extra_cols = self.test_df_list[0].columns.difference(self.DM.coords+self.DM.vars)
        for test_df in self.test_df_list:
            input = jnp.array(test_df[self.DM.coords].to_numpy())
            pred = self.model.apply(self.params, input)
            pred_df = pd.DataFrame(np.asarray(pred), columns=self.DM.vars)
            unscale_pred_df = self.DM.unscale_outputs(pred_df)
            unscale_test_coord_df = self.DM.unscale_inputs(test_df[self.DM.coords])
            unscale_pred_df = pd.concat([unscale_test_coord_df, unscale_pred_df, test_df[extra_cols].reset_index()], axis=1)
            pred_df_list.append(unscale_pred_df)
        return pred_df_list
    
    def _predict_complete_set(self):
        pred_df_list = []
        extra_cols = self.df_all_scaled_list[0].columns.difference(self.DM.coords+self.DM.vars)
        for df in self.df_all_scaled_list:
            input = jnp.array(df[self.DM.coords].to_numpy())
            pred = self.model.apply(self.params, input)
            pred_df = pd.DataFrame(np.asarray(pred), columns=self.DM.vars)
            unscale_pred_df = self.DM.unscale_outputs(pred_df)
            unscale_test_coord_df = self.DM.unscale_inputs(df[self.DM.coords])
            unscale_pred_df = pd.concat([unscale_test_coord_df, unscale_pred_df, df[extra_cols].reset_index()], axis=1)
            pred_df_list.append(unscale_pred_df)
        return pred_df_list
    
    def _predict_MSE_pinn(self):

        def NN(params, z, r, TI, CT):
            inp = jnp.array([z, r, TI, CT])
            func = self.model.apply(params, inp)
            return func
        
        def data_loss_fn(params, pred, true):
            "Dummy function"
            return 1
        
        def simpler_pinn_loss(colloc, grid_data, flow_data):
            self.predictions_list = self._predict_test_set()
            _, MSE_pinn = data_and_pinn_loss_non_jit(self.params, colloc, grid_data, flow_data, self.DM.non_dim_vals, self.DM.coords_min_max, self.DM.vars_min_max, NN, data_loss_fn)
            return MSE_pinn
        
        MSE_pinn_list = []
        for test_df in self.test_df_list:
            colloc = jnp.array(test_df[self.DM.coords].to_numpy())
            grid_data = jnp.array(test_df[self.DM.coords].to_numpy())
            flow_data = jnp.array(test_df[self.DM.vars].to_numpy())
            MSE_pinn = simpler_pinn_loss(colloc, grid_data, flow_data)
            MSE_pinn_list.append(MSE_pinn)

        self.MSE_pinn_list = MSE_pinn_list
        MSE_pinn_tot = np.mean(np.array(MSE_pinn_list))
        return MSE_pinn_tot
    
    def plot_flow_contour(self, plot_vals, var, plot_type=None, x_label=True, y_label=True, ax=None, use_mask=True, cbar_orientation="vertical"):

        if var == 'P':
            cbar_units = r' [Pa/$\rho u_\infty^2$]'
        else:
            cbar_units = r' [(m/s)/$u_\infty$]'
        
        # if plot_type == 'true':
        #     cbar_label = f"${var.lower()}$"+cbar_units
        #     oom = 1
        if (plot_type == 'pred' or plot_type == 'true'):
            if var == "U_z":
                oom = 0
                cbar_label = f"${var.lower()}$"
            else:
                oom = -2
                cbar_label = f"${var.lower()} \\times 10^{{{oom}}}$"
        elif plot_type == 'err':
            oom = -3
            if var == "P":
                var_ref = f"{var.lower()}" + r"_{\mathrm{ref}}"
            else:
                var_split = var.lower().split("_")
                var_ref = var_split[0] + r"_{" + f"{var_split[1]}" + r",\mathrm{ref}}"
            cbar_label = f"$\|{var.lower()} - {var_ref}\| \\times 10^{{{oom}}}$"


        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 2))

        plot_vals = plot_vals.reshape(self.grid_shape)
        if use_mask:
            plot_vals[self.exclude_mask] = np.nan
        cf = ax.contourf(self.z_grid, self.r_grid, plot_vals)
        if x_label:
            ax.set_xlabel(r'$z$ [$m/D$]')
        if y_label:
            ax.set_ylabel(r'$r$ [$m/D$]')
        divider = make_axes_locatable(ax)
        if cbar_orientation == "vertical":
            cax = divider.append_axes('right', size='5%', pad=0.05)
        elif cbar_orientation == "horizontal":
            if var == 'P':
                cax = divider.append_axes('bottom', size='10%', pad=0.6)
            else:
                cax = divider.append_axes('bottom', size='10%', pad=0.2)
    
        cbar = plt.colorbar(cf, cax=cax, orientation=cbar_orientation,  format=OOMFormatter(oom, fformat="%.2f", mathText=True)) #, format="%.2e")
        if cbar_orientation == "horizontal":
            cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
            cbar.ax.set_xlabel(cbar_label, size=16)
        else:
            cbar.ax.set_ylabel(cbar_label, size=16)
        return ax
    
    def plot_test_flow_contour(self, var, flowcase=0, x_label=True, y_label=True, ax=None, use_mask=True, cbar_orientation="vertical", use_full_fc_set=False) :
        if use_full_fc_set:
            plot_df = self.unscaled_df_all[flowcase]
        else:
            plot_df = self.unscaled_test_set[flowcase]
        ax = self.plot_flow_contour(plot_df[var].to_numpy(), var, plot_type="true", x_label=x_label, y_label=y_label, ax=ax, use_mask=use_mask, cbar_orientation= cbar_orientation)
        return ax
    
    def plot_model_pred_contour(self, var, flowcase=0, x_label=True, y_label=True, ax=None, use_mask=True, cbar_orientation="vertical",use_full_fc_set=False):
        if use_full_fc_set:
            plot_df = self.df_all_predictions_list[flowcase]
        else:
            plot_df = self.predictions_list[flowcase]
        ax = self.plot_flow_contour(plot_df[var].to_numpy(), var, plot_type="pred", x_label=x_label, y_label=y_label, ax=ax, use_mask=use_mask, cbar_orientation= cbar_orientation)
        return ax

    def plot_err_contour(self, var, flowcase=0, x_label=True, y_label=True, ax=None, use_mask=True, cbar_orientation="vertical", use_full_fc_set=False):
        if use_full_fc_set:
            plot_df = (self.unscaled_df_all[flowcase][var]-self.df_all_predictions_list[flowcase][var]).abs()
                #    /self.unscaled_df_all[flowcase][var])*100
        else:
            plot_df = (self.unscaled_test_set[flowcase][var]-self.predictions_list[flowcase][var]).abs()
                #    /self.unscaled_test_set[flowcase][var])*100
        ax = self.plot_flow_contour(plot_df.to_numpy(), var, plot_type="err", x_label=x_label, y_label=y_label, ax=ax, use_mask=use_mask, cbar_orientation= cbar_orientation)
        return ax

    def plot_pred_triplet(self, variable, flowcase=0):

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes = axes.flatten()
        axes[0] = self.plot_test_flow_contour(variable, flowcase=flowcase, x_label=False, y_label=True, ax=axes[0])
        axes[1] = self.plot_model_pred_contour(variable, flowcase=flowcase, x_label=False, y_label=True, ax=axes[1])
        axes[2] = self.plot_err_contour(variable, flowcase=flowcase, x_label=True, y_label=True, ax=axes[2])
        return fig, axes
    
    def plot_wake_centerline(self, z_depth=10, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for pred_df, true_df in zip(self.df_all_predictions_list, self.unscaled_df_all):
            pred_df = pred_df[pred_df['z_cyl'] == z_depth]
            true_df = true_df[true_df['z_cyl'] == z_depth]
            ax.plot(pred_df['r'], pred_df['U_z'], label='Prediction')
            ax.plot(true_df['r'], true_df['U_z'], label='True')
        ax.set_xlabel(r'$U_z$')
        ax.set_ylabel(r'$r$ [$D$]')
        ax.legend()
        return ax


    def plot_tensorboard_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        _, epochs_data, loss_data = self.TB_loader.load_scalar_events("Loss/data")
        _, _, val_losses = self.TB_loader.load_scalar_events("Loss/val")

        loss_list = [loss_data, val_losses]
        label_list = ['Data', 'Validation']
        
        try: 
            _, _, pinn_losses = self.TB_loader.load_scalar_events("Loss/pinn")
            loss_list.append(pinn_losses)
            label_list.append('PINN')
            loss_list.append(loss_data+pinn_losses)
            label_list.append('Total')
        except: pass

 
        for loss, label in zip(loss_list, label_list):
            ax.semilogy(epochs_data, loss, label=label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.set_xlabel('Epoch [-]')
        ax.set_ylabel(r'$\mathcal{L}$')
        return ax
    
    def make_metric_df(self, test_special_flowcase=None):
        model_vals = pd.DataFrame(np.array(list(self.cfg['model'].values())).reshape(-1,1).T, columns=list(self.cfg['model'].keys()))

        if test_special_flowcase is not None:
            """This is a special case for testing only"""
            print("WARNING YOU ARE ONLY TESTING ON A SPECIAL FLOWCASE POTENTIALLY NOT FROM THE TESTSET")
            print("to disable this remove the optional flag: test_special_flowcase in the make_metric_df method")
            total_test_df = self.unscaled_df_all[test_special_flowcase]
            total_test_df = total_test_df[self.DM.vars]
            total_pred_df = self.df_all_predictions_list[test_special_flowcase][self.DM.vars]
        else:
            total_test_df = pd.DataFrame()
            for flowcase in range(len(self.unscaled_test_set)):
                test_df = self.unscaled_test_set[flowcase]
                test_df = test_df[self.DM.vars]
                total_test_df = pd.concat([total_test_df, test_df], axis=0)

            total_pred_df = pd.DataFrame()
            for flowcase in range(len(self.predictions_list)):
                pred_df = self.predictions_list[flowcase][self.DM.vars]
                total_pred_df = pd.concat([total_pred_df, pred_df], axis=0)

        
        err_df = total_test_df - total_pred_df
        metrics = pd.DataFrame([], columns=["MSE", "MAE", "RMSE", "MAPE", "R2", "MSE_pinn"], index=self.DM.vars+["total"])
        for var in self.DM.vars:
            err = total_test_df[var] - total_pred_df[var]
            MSE = np.mean(err**2)
            MAE = np.mean(np.abs(err))
            RMSE = np.sqrt(MSE)
            MAPE = np.mean(np.abs(err/total_test_df[var]))
            R2 = 1 - np.sum(err**2)/np.sum((total_test_df[var] - np.mean(total_test_df[var]))**2) 
            metrics.loc[var] = np.array([MSE, MAE, RMSE, MAPE, R2, np.nan]).reshape(1,-1)
        
        flat_err = err_df.dropna().values.flatten()
        flat_test = total_test_df.dropna().values.flatten()

        MSE_tot = np.mean(flat_err**2)
        MAE_tot = np.mean(np.abs(flat_err))
        RMSE_tot = np.sqrt(MSE_tot)
        MAPE_tot = np.mean(np.abs(flat_err/flat_test))
        SSR_tot = np.sum(flat_err**2)
        SST_tot = np.sum((flat_test - np.mean(flat_test))**2)
        R2_tot = 1 - SSR_tot/SST_tot
        MSE_pinn_tot = self.MSE_pinn
        metrics.loc["total"] = np.array([MSE_tot, MAE_tot, RMSE_tot, MAPE_tot, R2_tot, MSE_pinn_tot]).reshape(1,-1)
        return metrics, model_vals

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from utils.data import DataManager

    DM = DataManager(nc_path = '~/code/jax-flax-wake-pinn/Data/RANS_1wt_irot_v2.nc', 
                     exclusion_radius = 1., 
                     input_coords = ["z_cyl", "r", "CT", "TI_amb"],
                     output_vars = ["U_z", "U_r", "P"], 
                     val_split=0.1, 
                     development_mode=False)

    model_path = '../Results/Experiment2/14/checkpoints/1173/default'
    plotter_class = Plotter(DM, model_path)
    plotter_class.make_metric_df()

    flowcase = 0
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    for ax, var in zip(axes, ['U_z', 'U_r', 'P']):
        ax = plotter_class.plot_test_flow_contour(var, flowcase=flowcase, ax=ax)
        ax.axis('equal')
    plt.suptitle(f'Flowcase {flowcase}') # TODO add an interepation of flowcase i.e. TI and CT
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    for ax, var in zip(axes, ['U_z', 'U_r', 'P']):
        ax = plotter_class.plot_model_pred_contour(var, flowcase=flowcase, ax=ax)
        ax.axis('equal')
    plt.suptitle(f'Flowcase {flowcase}') # TODO add an interepation of flowcase i.e. TI and CT
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes = axes.flatten()
    for ax, var in zip(axes, ['U_z', 'U_r', 'P']):
        ax = plotter_class.plot_err_contour(var, flowcase=flowcase, ax=ax)
        ax.axis('equal')
    plt.suptitle(f'Flowcase {flowcase}') # TODO add an interepation of flowcase i.e. TI and CT
    plt.show()

    variable_list = ['U_z', 'U_r', 'P']
    flowcase_list = [0, 1, 2]
    for flowcase in flowcase_list:
        for variable in variable_list:
            fig, axes = plotter_class.plot_pred_triplet(variable, flowcase=flowcase)
            plt.suptitle(f'Flowcase: {flowcase}')
            plt.show()
    
    fig, ax = plt.subplots()
    ax = plotter_class.plot_tensorboard_loss(ax=ax)
    plt.show()