# Cluster baseline

The cluster uses the existing environment at `/home/fao8402/venvs/thesis` and the repository at
`/home/fao8402/t1t2_component_detection`.

Generate the 100k baseline data once if it is missing:

```bash
sbatch slurm/gen_data.slurm
```

Train the baseline:

```bash
sbatch slurm/train.slurm
```

Watch the job:

```bash
squeue -u "$USER"
tail -f slurm/logs/train_t1t2_train_*.out
```

The run uses `configs/cluster.yaml`. It trains on 99,999 balanced voxels with 64 inputs and
`n_comp=1..3`. If the 24-hour job limit is reached, submit `train.slurm` again; training resumes
from `last.pt`.

Audit notebooks, smoke jobs, tests, and scaling experiments are optional later work. They do not
block this baseline.
