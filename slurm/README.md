# Running `t1t2_training` on the HPC (Slurm) cluster

The whole folder is self-contained (it vendors its own data generator), so the workflow is:
**copy it up → generate data on the cluster → train on a GPU node.**

> ⚠️ **Fill in the cluster-specific bits first.** Every `<...>` placeholder in the `.slurm`
> files and `setup_env.sh` (partition names, account, module names) comes from your HPC's
> "How To" guide. Search this folder for `<` to find them all. Nothing here is guessed to be
> correct for your specific cluster — only the structure is.

## 0. Copy the project to the cluster

```bash
# from your Mac (exclude the big local data/results — we regenerate data on the cluster)
rsync -av --exclude 'data/' --exclude 'results/' --exclude '__pycache__' \
      --exclude '.pytest_cache' ~/Desktop/Thesis/t1t2_training/ \
      <USER>@<CLUSTER_HOST>:~/t1t2_training/
```

## 1. One-time environment setup (on a login node)

```bash
cd ~/t1t2_training
bash slurm/setup_env.sh        # creates a venv and installs the CUDA torch + deps
```

## 2. Generate the full dataset (CPU job — keeps the big files on the cluster)

```bash
sbatch slurm/gen_data.slurm    # writes data/full/{train,val,test,test_snr*}.parquet
```

## 3. Train (GPU job)

```bash
sbatch slurm/train.slurm       # trains configs/cluster.yaml -> results/<name>/
# watch it:
squeue -u $USER
tail -f slurm/logs/t1t2-train_*.out
```

## Notes
- `--device` is left `null` in the config → auto-detects **cuda** on the GPU node.
- The Hungarian loss does per-voxel matching on CPU, so request a few CPU cores per GPU task
  (`--cpus-per-task`) — the matmuls run on the GPU, the matching on those cores.
- Checkpoints resume automatically: re-submitting `train.slurm` continues from `last.pt`, which
  is what makes it safe against the wall-clock `--time` limit (bump epochs and re-submit).
