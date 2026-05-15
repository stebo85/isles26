## general guidelines
- no credentials in the repo
- commit and push after new features are implemented
- when learning new insights make sure to update agents.md with a short description and longer documentation goes in docs (linked from agents.md)
- literature review for ISLES'26 preparation lives in [docs/README.md](docs/README.md); key current lessons are to start with nnU-Net/MAPPING-style baselines, build metrics first, stratify validation by center/chronicity/lesion size, and treat small-lesion detection as the main risk, and compare any attention/transformer/SAM branch against nnU-Net under identical splits
- don't run large analyses directly - write and sumbit slurm job files! Make sure to make them robust to interruption

```bash
#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -p owners


module purge
module use $GROUP_HOME/modules/
module load ants/2.6.0
ants.... $1
```

and submit them to the owners que:
```bash
sbatch -p owners submit.sbatch
```
