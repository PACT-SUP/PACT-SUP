# PACT: paper's supplement

Code, checkpoints and data to recompute every PACT number in the paper, and to retrain the model.

| file | what it holds |
|---|---|
| `pact/model.py` | the model |
| `pact/train.py` | the trainer |
| `pact/paper.py` | one function per table or paragraph of the paper |
| `run.py` | `verify`, `report`, `train`; compares each printed number with its recomputed value |
| `checkpoints/shipped/` | `lp0`-`lp5`: LP-PDBbind seeds 0-5; `cleansplit0`-`cleansplit4`: CleanSplit folds 0-4 |
| `expected/paper_values.json` | the 595 printed numbers, each with its paper line and where it is recomputed (extracted from the paper with Claude Code) |

## Quickstart

NOTE: The commands below assume a Linux shell, Windows may need workarounds.

Download the data archive `pact-data-v1.tar` (3.4 GiB) from
[MEGA](https://mega.nz/file/UVhEWBqD#mZew9bf1Xy35wVF5vsm3Owv_-yvU4wrdVGC78WUNZ00), then:

```bash
conda env create -f environment.yml && conda activate pact
tar -xf ../pact-data-v1.tar -C ..       # unpacked next to this folder
python fetch_data.py --verify          # every data file against SHA256SUMS
python run.py verify                   # CPU only: out/shipped/report.md
```

Data is read from `$PACT_DATA`, which defaults to `../pact-data-v1`. `verify` first checks that
every data file and checkpoint is present. If any is missing, it stops with their names.

Retraining needs a CUDA GPU. It retrains the model into `checkpoints/retrained/`, then checks the paper
again:

```bash
python run.py train                          # the 6 LP seeds
python run.py train --cleansplit             # the 5 CleanSplit folds (much heavier)
python run.py verify --checkpoints retrained # out/retrained/report.md
```

The CleanSplit folds only give the CASF-2016 Pearson r (mean, SD and the two differences built on it).
Without them, `verify` checks everything else and reports those rows as **not run**.

`run.py verify --only accuracy rigid` runs chosen parts (the keys of `PORTS`). `run.py report`
rebuilds the report from saved outputs.

## Hardware and times (measured)

| step | hardware | wall clock |
|---|---|---|
| `verify` (all 595 numbers) | AMD EPYC 7763, CPU only | 7 min |
| `train --lp` (6 runs at once) | one RTX A4000 (16 GB), driver 610.57, CUDA 13.0, torch 2.12.1 | 25 min |
| `train --cleansplit` (5 runs at once) | a second RTX A4000 | 81 min |

## Reading `report.md`

Each row is one printed number: its table or section, its line in the paper source, the printed
token, the recomputed value, and the part of `pact/paper.py` that recomputes it.
- **PASS**: the recomputed value, rounded to the printed precision, equals the printed token.
  Bounds such as "under 0.03" must hold as stated, and "about 60" must round to 60.
- **FAIL**: it does not.
- **error**: its part failed (the traceback is printed during `verify`).
- **not run**: the part was excluded by `--only`, or the row needs the CleanSplit folds and they
  were not retrained.

With the shipped checkpoints every row passes.

**Retraining and byte identity.** All eleven shipped checkpoints were trained with this trainer.
Retraining them on the GPUs above reproduced every one byte for byte, and the report then uses PASS
and FAIL as for the shipped ones. Other GPUs, drivers or library versions can change low decimals.
When any retrained checkpoint differs, the report labels each row:
- **consistent** when the value lies within twice its cross-seed SD, or within 5% for a number
  that is not a seed mean;
- **differs** otherwise.


## What is covered

Every PACT number in the main text and the appendix, in tables, prose and captions:
- accuracy on LP-PDBbind, CASF-2016 (CleanSplit) and the external sets, and the ligand-size control;
- seed stability, and rigid-body and protonation robustness;
- faithfulness: interventions (native and occlusion), single contacts, deletion curves, selection
  fractions, sparsity;
- chemistry: agreement with ProLIF, pose recovery, 3D matched molecular pairs, coefficient use;
- the worked example and the model size.


## Data provenance and licensing

`pact-data-v1` holds preprocessed graphs and labels, not raw structures:
- derived from PDBbind: the LP-PDBbind split, PDBbind CleanSplit with CASF-2016, and the external
  evaluation sets BDB2020+, Mpro and EGFR, as described in the paper;
- also: ProLIF interaction labels for the test split, redocked CASF-2016 poses, 3D matched
  molecular pairs, and protonation-state variants.

PDBbind-derived files are provided only to reproduce the paper's results. Using PDBbind itself is
subject to its own licence terms. The code is released for review, a licence will come with the public
release.
