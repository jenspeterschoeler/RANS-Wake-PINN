#%% Imports
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logger = logging.getLogger(__name__)

class TensorBoardLoader:
    def __init__(self, path):
        self.path = path
        self.event_acc = EventAccumulator(self.path)
        self.event_acc.Reload()
        self.tags = self.event_acc.Tags()
    
    def load_scalar_events(self, scalar_name):
        data = self.event_acc.Scalars(scalar_name)
        w_times = []
        step_nums = []
        vals = []
        for scalar in data:
            w_times.append(scalar.wall_time)
            step_nums.append(scalar.step)
            vals.append(scalar.value)
        return w_times, step_nums, vals

    def load_df(self):
        tags = self.tags["scalars"]
        initial_tag = tags.pop(0)
        w_times, step_nums, vals = self.load_scalar_events(initial_tag)
        df = pd.DataFrame({"w_times": w_times, "step": step_nums, "loss": vals})
        for tag in tags:
            w_times, step_nums, vals = self.load_scalar_events(tag)
            len_diff = len(df) - len(vals)
            if len_diff:
                nan_insert = np.full(len_diff, np.nan)
                vals = np.concatenate([nan_insert, vals])
            df[tag] = vals
        return df

    # def load_df(self):
    #     tags = self.tags["scalars"]
    #     initial_tag = tags.pop(0)
    #     w_times, step_nums, vals = self.load_scalar_events(initial_tag)
    #     df = pd.DataFrame({"w_times": w_times, "step": step_nums, "loss": vals})
    #     for tag in tags:
    #         w_times, step_nums, vals = self.load_scalar_events(tag)
    #         df[tag] = vals
    #     return df


class DataLoader:
    def __init__(self, X, y, org_train_grid_shape, batch_size, physics_on, data_ratio, colloc_data_ratio, train_min_max, exclusion_radius, rng_key):
        self.train_min_max = train_min_max
        
        self.pointer = 0
        self.pointer_c = 0
        self.rkey = rng_key


        self.batch_size = batch_size

        n_max = len(X)
        self.data_ratio = data_ratio

        if self.data_ratio != 1:
            rkey, subkey = jax.random.split(self.rkey)
            self.rkey = rkey
            org_indices = jnp.arange(len(X))
            idx_keep = jax.random.choice(subkey, org_indices, shape=(int(jnp.floor(n_max*data_ratio)),), axis=0)
            self.X = X[idx_keep, :]
            self.y = y[idx_keep, :]
        else:
            self.X = X
            self.y = y
        self.indices = jnp.arange(len(self.X))

        self.n_training = int(jnp.floor(n_max*data_ratio))

        self.physics_on = physics_on
        if self.physics_on:
            self.colloc_data_ratio = colloc_data_ratio
            self.n_colloc = int(jnp.floor(n_max*self.colloc_data_ratio))
            self.batch_size_c =  int(jnp.floor(self.batch_size*(self.n_colloc/self.n_training)))

        self.exclusion_radius = exclusion_radius
        self.n_batches = int((len(self.X) - len(self.X)%self.batch_size)/self.batch_size)

        self.org_train_grid_shape = org_train_grid_shape
        self.num_zr = jnp.floor(jnp.array(self.org_train_grid_shape[["z_cyl", "r"]])*2.5).astype(int) #HACK hardcoded the 2.5 factor for colloc grid ratio

        colloc_zr = self._get_minmax_scaled_grided_zr_colloc()
        #  colloc_TI, colloc_CT
        self.X_c_zr = colloc_zr
        self.X_c_zr_idx = jnp.arange(self.X_c_zr.shape[0])
        # TODO cleanup these if statements perhaps with a function
        if self.org_train_grid_shape["TI_amb"] == 1:
            self.X_c_TI_range = jnp.array([0, 0])
        else:
            self.X_c_TI_range = jnp.array([-1, 1])

        if self.org_train_grid_shape["CT"] == 1:
            self.X_c_CT_range = jnp.array([0, 0])
        else: 
            self.X_c_CT_range = jnp.array([-1, 1])
        print("================================")
        print(f"Dataloader loaded with:\n \t n_batches = {self.n_batches}")
        print("================================")
    
    def __iter__(self):
        return self

    def __next__(self):
        
        # check if iterator is empty
        end_point = self.pointer+self.batch_size
        if end_point >= len(self.X):
            self.pointer = 0    # Reset the pointer
            raise StopIteration

        # Prepare the  data compoentn of the batch
        batch_indices = self.indices[self.pointer:end_point]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        self.pointer += self.batch_size # Update the pointer between batches

        if self.physics_on:
            rkey, subkey1, subkey2, subkey3 = jax.random.split(self.rkey, 4)
            self.rkey = rkey
            X_batch_zr_c_idx = jax.random.choice(subkey1, self.X_c_zr_idx, shape=(self.batch_size_c,), axis=0)
            X_batch_zr_c = self.X_c_zr[X_batch_zr_c_idx, :]
            X_batch_TI_c = jax.random.uniform(subkey2, shape=(self.batch_size_c, 1), minval=self.X_c_TI_range[0], maxval=self.X_c_TI_range[1])
            X_batch_CT_c = jax.random.uniform(subkey3, shape=(self.batch_size_c, 1), minval=self.X_c_CT_range[0], maxval=self.X_c_CT_range[1])
            X_batch_c = jnp.concatenate((X_batch_zr_c, X_batch_TI_c, X_batch_CT_c), axis=1)

            print(f"shapes: X_batch: {X_batch.shape}, y_batch: {y_batch.shape}, X_batch_c: {X_batch_c.shape}")
            return X_batch, y_batch, X_batch_c
        else:
            return X_batch, y_batch


    def _get_minmax_scaled_grided_zr_colloc(self):
        """Assumes min max scaler"""
        def min_max_scaled_uniform_data(num_points):
            if num_points == 1: # when there is a single point, the linspace function will return -1, but we want 0. This is because we are skipping a step by not first generating the data in the unscaled space and then scaling it.
                colloc_array = jnp.array([0])
            else:
                colloc_array = jnp.linspace(-1, 1, num_points)
            return colloc_array

        z = min_max_scaled_uniform_data(self.num_zr[0])
        r = min_max_scaled_uniform_data(self.num_zr[1])
        # colloc_TI = min_max_scaled_uniform_data(self.num_zrTICT[2])
        # colloc_CT = min_max_scaled_uniform_data(self.num_zrTICT[3])

        z_grid, r_grid = jnp.meshgrid(z, r, indexing='ij')

        z_grid = z_grid.ravel()
        r_grid = r_grid.ravel()

        if self.exclusion_radius is not None:
            r_grid_unscaled = self._min_max_unscale(r_grid, "r")
            z_grid_unscaled = self._min_max_unscale(z_grid, "z_cyl")
            idx_keep = np.sqrt(r_grid_unscaled**2+z_grid_unscaled**2) > self.exclusion_radius
            z_grid = z_grid[idx_keep]
            r_grid = r_grid[idx_keep]

        z_grid = jnp.expand_dims(z_grid, 1)
        r_grid = jnp.expand_dims(r_grid, 1)

        colloc_zr = jnp.concatenate([z_grid, r_grid], axis=1)
        return colloc_zr
    
    def _min_max_unscale(self, data, coord_name):
        # TODO Check if this is correct
        data_unscaled = ((data+1)/2)*(self.train_min_max[coord_name][1]- self.train_min_max[coord_name][0]) + self.train_min_max[coord_name][0]
        return data_unscaled

    def shuffle_data(self):
        self.indices = jax.random.permutation(self.rkey, self.indices)
        self.X = self.X[self.indices]
        self.y = self.y[self.indices]


class DataManager:
    """Class to handle the data from the netcdf file
    """
    def __init__(self, nc_path: str, exclusion_radius: float | None = None, input_coords: list = ['z_cyl', 'r', 'TI_amb', 'CT'],  output_vars: list = ['U_z', 'U_r', 'P'], val_split: float = 0.1, development_mode=False):
        self.nc_path = nc_path
        self.exclusion_radius = exclusion_radius
        self.coords = input_coords
        self.vars = output_vars
        self.val_split = val_split
        self.development_mode = development_mode

        self.ds = xr.open_dataset(self.nc_path, engine='netcdf4')
        self.coords_min_max_full_ds = self._get_min_max_ds(self.coords)
        self.vars_min_max_full_ds = self._get_min_max_ds(self.vars)
        all_vars = [var for var in self.ds.data_vars]
        self.drop_vars = list(set(all_vars) - set(self.vars))
        self.ds = self.ds.drop_vars(self.drop_vars)
        self.non_dim_vals = self._get_non_dim_values()
        self.ds = self._interpolate()
        self.data_columns = self.coords + self.vars # exclusion_mask and flowcase are added in _append_exclusion_mask and _append_train_test_flowcase_indexes
        self._append_exclusion_mask()
        self._append_train_test_flowcase_indexes()
        self.fc_all = np.append(self.train_fc, self.test_fc, axis=1)
        self.df_all = self._get_df(self.fc_all)
        self.coords_min_max = self._get_min_max_df(self.df_all, self.coords)
        self.vars_min_max= self._get_min_max_df(self.df_all, self.vars)
        self.df_train = self._get_df(self.train_fc)
        self.org_train_grid_shape = self.df_train[self.coords].nunique()
        self.df_test = self._get_df(self.test_fc)
        df_train_scaled, min_max_array = self._scale_data(self.df_train)
        self.min_max_array = min_max_array
        self.df_train_scaled, self.df_val_scaled = self._train_val_split(df_train_scaled)
        self.df_test_scaled, _ = self._scale_data(self.df_test)
        self.df_all_scaled, _ = self._scale_data(self.df_all)

    def _get_non_dim_values(self):
        non_dim_values = {}
        for key in self.ds.attrs:
            current_type = type(self.ds.attrs[key])
            if np.issubdtype(current_type, float) or np.issubdtype(current_type, int):
                non_dim_values[key] = jnp.float32(self.ds.attrs[key])
        logger.info("Some non-dimensionalization values are hardcoded")
        #HACK hardcoded values (should be metadata from in the NetCDF file from PyWakeEllipsys)
        non_dim_values["C_mu"] = jnp.float32(0.03)             # Turbulence model constant
        non_dim_values['mu'] = jnp.float32(1.78406e-05)        # kg/m/s (from PyWakeEllipsys docs)
        non_dim_values['z_ref'] = jnp.float32(119.0)           # hub height DTU10MW
        return non_dim_values


    def _get_min_max_ds(self, keys):
        min_max = {}
        for key in keys:
            min_max[key] = (self.ds[key].min().values, self.ds[key].max().values)
        return min_max

    def _get_min_max_df(self, df, keys):
        min_max = {}
        for key in keys:
            min_max[key] = (df[key].min(), df[key].max())
        return min_max

    def _interpolate(self):
        logger.info("Dataset is interpolated with a hardcoded distance")
        z_cyl_dist = np.diff(self.ds.z_cyl)[0]
        z_cyl_interp = np.arange(self.coords_min_max_full_ds['z_cyl'][0], self.coords_min_max_full_ds['z_cyl'][1], z_cyl_dist)
        ds_interp = self.ds.interp(z_cyl=z_cyl_interp)
        return ds_interp

    def _append_exclusion_mask(self):
        z_grid, r_grid = np.meshgrid(self.ds.z_cyl, self.ds.r, indexing='ij')
        if self.exclusion_radius == None:
            self.exclusion_radius = 0
        exclusion_mask = np.sqrt(z_grid**2 + r_grid**2) < self.exclusion_radius 
        self.ds['exclusion_mask'] = (('z_cyl', 'r'), exclusion_mask)
        self.data_columns = self.data_columns + ['exclusion_mask']
    
    def _append_train_test_flowcase_indexes(self):
        CT_grid, TI_grid = np.meshgrid(self.ds.CT, self.ds.TI_amb, indexing='ij')
        if self.development_mode == "subset3x3":
            CT_grid = CT_grid[1:4, 1:4]
            TI_grid = TI_grid[1:4, 1:4]
            CT_test = CT_grid[1, 1]
            TI_test = TI_grid[1, 1]
            CT_train = np.copy(CT_grid)
            CT_train[1, 1] = np.nan
            TI_train = np.copy(TI_grid)
            TI_train[1, 1] = np.nan
        else:
            # Calculate the diagonal indices
            start = (0, 0)  # starting point (row, column)
            direction = (1, 1)  # direction (d_row, d_column)
            diag_indices = [(start[0]+i*direction[0], start[1]+i*direction[1]) for i in range(min(CT_grid.shape))]
            diag_indices = tuple(np.array(list(t)) for t in zip(*diag_indices))
            diag_indices[0][0] = 1 # move one to not be on the edge of the domain

        
            CT_test = CT_grid[diag_indices]
            TI_test = TI_grid[diag_indices]

            CT_train = np.copy(CT_grid)
            CT_train[diag_indices] = np.nan

            TI_train = np.copy(TI_grid)
            TI_train[diag_indices] = np.nan

            if self.development_mode == "singleCT":
                CT_idx = CT_train.shape[0]//2
                CT_train = CT_train[CT_idx, :]
                TI_train = TI_train[CT_idx, :]
                assert len(np.unique(CT_train[~np.isnan(CT_train)])) == 1, f"Unique CT's: {CT_train[~np.isnan(CT_train)]}\nCT_train should only contain one unique value"
                CT_test_idx = np.where(CT_test == CT_train[0])
                CT_test = CT_test[CT_test_idx]
                TI_test = TI_test[CT_test_idx]

        CT_train = CT_train.flatten()
        CT_train = CT_train[~np.isnan(CT_train)]      
        
        TI_train = TI_train.flatten()
        TI_train = TI_train[~np.isnan(TI_train)]
        
        self.test_fc = np.vstack([CT_test, TI_test])
        self.train_fc = np.vstack([CT_train, TI_train])
        self.data_columns = self.data_columns + ['flowcase']

    def _get_df(self, flowcases):
        df = pd.DataFrame()
        for i, (CT_i, TI_i) in enumerate(np.split(flowcases, flowcases.shape[-1], axis=1)):
            ds_fc = self.ds.sel(CT=[CT_i], TI_amb=[TI_i]) 
            df_fc = ds_fc.to_dataframe()
            df_no_index_fc = df_fc.index.to_frame().reset_index(drop=True)
            for col in list(df_fc.columns):
                df_no_index_fc[col] = df_fc[col].to_numpy()
            df_no_index_fc['flowcase'] = i
            df = pd.concat([df, df_no_index_fc], ignore_index=True)
        assert len(self.data_columns) == len(df.columns), f"Number of columns does not match assumed for hardcoded data pre-process. The columns are: {df.columns}" 
        df = df[self.data_columns]
        return df

    def _scale_data(self, df):
        def min_max_scale(data, min_max):
            return 2*((data - min_max[:, 0])/(min_max[:, 1]- min_max[:, 0]))-1
        scale_columns = self.coords + self.vars
        min_max_array_coord = np.array([self.coords_min_max[coord] for coord in self.coords])
        min_max_array_vars = np.array([self.vars_min_max[var] for var in self.vars])
        min_max_array = np.vstack([min_max_array_coord, min_max_array_vars])

        res = min_max_scale(df[scale_columns], min_max_array)
        extra_columns = list(set(df.columns) - set(scale_columns))
        df_out = pd.concat([res, df[extra_columns]], axis=1)
        return df_out, min_max_array

    def _train_val_split(self, df):
        df_train = df.sample(frac=1-self.val_split, random_state=123) #HACK always same val/trian split, this could be optional # TODO make this reliant on jax random key instead of pandas
        df_val = df.drop(df_train.index)
        return df_train, df_val

    def get_train_dataloader(self, batch_size, physics_on, data_ratio, colloc_data_ratio, rng_key=jax.random.PRNGKey(123)):
        self.batch_size = batch_size
        df_train = self.df_train_scaled

        self.data_ratio = data_ratio
        self.colloc_data_ratio = colloc_data_ratio

        data_loader = DataLoader(jnp.array(df_train[self.coords].to_numpy()), 
                                 jnp.array(df_train[self.vars].to_numpy()),
                                 org_train_grid_shape=self.org_train_grid_shape, 
                                 batch_size=batch_size,
                                 physics_on=physics_on,
                                 data_ratio=self.data_ratio,
                                 colloc_data_ratio=colloc_data_ratio,
                                 train_min_max=self.coords_min_max,
                                 exclusion_radius=self.exclusion_radius,
                                 rng_key=rng_key)
        return data_loader
    
    def get_validation_set(self):
        df_val = self.df_val_scaled
        X_val = jnp.array(df_val[self.coords].to_numpy())
        y_val = jnp.array(df_val[self.vars].to_numpy())
        return X_val, y_val

    def get_test_set(self):
        df = self.df_test_scaled
        test_df_list = []
        for flowcase in df['flowcase'].unique():
            df_fc = df[df['flowcase']==flowcase]
            test_df_list.append(df_fc)
        return test_df_list
    
    def get_complete_set(self):
        df = self.df_all_scaled
        complete_df_list = []
        for flowcase in df['flowcase'].unique():
            df_fc = df[df['flowcase']==flowcase]
            complete_df_list.append(df_fc)
        return complete_df_list


    def _min_max_unscale(self, data, min_max):
        return ((data+1)/2)*(min_max[:, 1]- min_max[:, 0]) + min_max[:, 0]

    def unscale_inputs(self, X):
        if type(X) == pd.DataFrame:
            X_df = X.copy()
            X_np = X_df[self.coords].to_numpy()
        else:
            X_np = X
            assert X_np.shape[-1] == len(self.coords)
        min_max_array_coord = self.min_max_array[:len(self.coords), :]
        X_np_unscaled = self._min_max_unscale(X_np, min_max_array_coord)
        if type(X) == pd.DataFrame:
            res_df = pd.DataFrame(X_np_unscaled, columns=self.coords)
            X_df = X_df.drop(columns=self.coords)
            X_df = X_df.reset_index(drop=True)
            res_df = res_df.reset_index(drop=True)
            X = pd.concat([res_df, X_df], axis=1)
        else:
            X = X_np_unscaled
        return X
 
    def unscale_outputs(self, y):
        if type(y) == pd.DataFrame:
            y_df = y.copy()
            y_np = y_df[self.vars].to_numpy()
        else:
            y_np = y
            assert y_np.shape[-1] == len(self.vars)
        min_max_array_vars = self.min_max_array[len(self.coords):, :]
        y_np_unscaled = self._min_max_unscale(y_np, min_max_array_vars)
        if type(y) == pd.DataFrame:
            res_df = pd.DataFrame(y_np_unscaled, columns=self.vars)
            y_df = y_df.drop(columns=self.vars)
            y_df = y_df.reset_index(drop=True)
            res_df = res_df.reset_index(drop=True)
            y = pd.concat([y_df, res_df], axis=1)
        else:
            y = y_np_unscaled
        return y

def get_data_manager(cfg):
    """Get the data manager from the config file
    """
    return DataManager(cfg.data.data_path, 
                       cfg.data.exclusion_radius,
                       cfg.data.input_coords,
                       cfg.data.output_vars,
                       cfg.optimizer.val_split,
                       cfg.data.development_mode)


if __name__ == "__main__":
    # Define the path to the netcdf file
    path = '~/code/jax-flax-wake-pinn/Data/RANS_1wt_irot_v2.nc'
    exclusion_radius = 1
    DM = DataManager(path, exclusion_radius, development_mode=False)


    train_data_loader = DM.get_train_dataloader(batch_size=100, physics_on=True, data_ratio=1, colloc_data_ratio=.5)
    # train_data_loader = DM.get_train_dataloader(batch_size=32, colloc_data_ratio=1)
    X_val, y_val = DM.get_validation_set()
    test_df_list = DM.get_test_set()

    if 0:
        DM.df_train.to_csv('./Data/train_2d_cyl.csv', index=False)
        DM.df_test.to_csv('./Data/test_2d_cyl.csv', index=False)


    for epoch in range(1):
        train_data_loader.shuffle_data()  # Shuffle the data at the beginning of each epoch
        i = 0
        X_batch_old = 0
        X_c_batch_old = 0
        for X_batch, y_batch, X_c_batch in train_data_loader:
            # Use X_batch and y_batch for training or evaluation
            print("X_batch_change:", X_batch- X_batch_old)
            print("y_batch:", y_batch)
            print("X_c_batch:", X_c_batch)
            print("X_c_batch_change:", X_c_batch-X_c_batch_old)
    
            X_batch_old = X_batch
            X_c_batch_old = X_c_batch
            i += 1
            if i > 3:
                break
        print(f"Batch shapes: X_batch: {X_batch.shape}, y_batch: {y_batch.shape}, X_c_batch: {X_c_batch.shape}")

    plot_df = test_df_list[0]
    plot_df_inputs = plot_df[DM.coords+["exclusion_mask"]]
    plot_df_inputs_unscaled = DM.unscale_inputs(plot_df_inputs)
    plot_df_outputs = plot_df[DM.vars]
    plot_df_outputs_unscaled = DM.unscale_outputs(plot_df_outputs)
    
    U_z = plot_df_outputs_unscaled['U_z'].to_numpy()
    U_z[plot_df_inputs_unscaled['exclusion_mask']] = np.nan
    grid_shape = (len(DM.ds.z_cyl), len(DM.ds.r))
    U_z = U_z.reshape(grid_shape)
    z_cyl = plot_df_inputs_unscaled['z_cyl'].to_numpy().reshape(grid_shape)
    r = plot_df_inputs_unscaled['r'].to_numpy().reshape(grid_shape)
    plt.figure(figsize=(6, 2))
    plt.contourf(z_cyl, r, U_z)
    plt.colorbar(orientation='horizontal')
    plt.axis('equal')
    plt.show();

# %%
