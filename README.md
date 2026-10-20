# PyPARIS_CoupledBunch_sim_class (Deprecated)
> [!CAUTION]
> This package has been fully integrated into PyECLOUD and it is installed alongside it, please refer to that repository:
> https://github.com/PyCOMPLETE/PyECLOUD
>
> **v1.0.2 is the last standalone release of PyPARIS_CoupledBunch_sim_class.** This repository is no longer maintained.
> Starting from PyECLOUD v8.9.0, PyPARIS_CoupledBunch_sim_class is distributed as part of PyECLOUD (reported as `1.0.2+pyec<PyECLOUD version>`),
> and all further development happens there. The import name is unchanged (`import PyPARIS_CoupledBunch_sim_class`).

## Instructions for running PyECLOUD-PyHEADTAIL coupled bunch simulations on CERN HPC cluster

* Create a workspace folder for software and simulations:
```
mkdir /hpcscratch/user/lmether/sim_workspace
```

* Load MPI: 
```
module load mpi/mvapich2/2.2
```

* Build our software stack in this folder by following the instructions here: https://github.com/PyCOMPLETE/PyECLOUD/wiki/Setup-python-3-%28including-mpi4py%29-without-admin-rights
* Clone this example into the simulation workspace
```
cd /hpcscratch/user/lmether/sim_workspace
git clone https://github.com/lmether/PyECLOUD_coupled_bunch_example
```
* Edit 000s_slurm_job to have the SIM_WORKSPACE variable point to your simulation workspace path

* Launch the simulation:
```
sbatch 000s_slurm_job
```

* Some code to display the simulation results can be found at: https://github.com/giadarol/CoupledBunchAnalysis
