# Plotting
Useful plot functions for the PET toolboxes. 

- plot_data: production data, RFT data, and (2d or 3d) seismic data
- plot_objective_function: data misfit, either all data combined or separated on well data and seismic data
- plot_parameters: layers of field parameters or surfaces, vertical averages of field parameters, scalars, or export field parameters to grid (as .grdecl files that can be visualized in e.g., ResInsight)
- plot_optim: objective function and state variables from popt

**Installation**

Inside the Plotting folder, run

    python3 -m pip install -e .

- The dot is needed to point to the current directory.
- The -e option installs PET such that changes to it take effect immediately (without re-installation).