# Lab 3 — a baseline and a first network

This folder is the project template with the plumbing filled in: a data
loader, a resize cache, a pixel table, seven handcrafted features, a
gradient-boosting baseline, Weights & Biases logging and the skeleton of a
training script. Six places are left for you, marked `TASK` in the source:
the grouped split, an MLP on raw pixels, the optimiser, the minibatch loop,
prediction and validation. The notebook `notebooks/lab3_baseline.ipynb` walks
through them in 90 minutes.

By the end of the lab this folder is your project repository, public on
GitHub, with the lab solved and committed. The project description has the
rules; the rest of this README has the infrastructure.

## Before the lab

Do these at home. In class they cost the hour.

**1. Docker and the environment.**

```bash
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
docker compose build
docker compose run --rm dev python scripts/check_env.py
```

If your machine has an NVIDIA GPU, uncomment the `COMPOSE_FILE` line for your
operating system in `.env`. Today's lab runs fine on a CPU.

**2. Weights & Biases.** Sign up or log in at https://wandb.ai, then copy your
API key from **https://wandb.ai/authorize** and paste it into `.env` after
`WANDB_API_KEY=`. Without a key your runs go to an anonymous account; that
works for today, not for the semester. Never commit `.env`.

**3. The training data.** It is a private Kaggle dataset, about 800 MB. The
link is on the lab slides and on Moodle. **Do not share the link and do not
re-upload the photographs anywhere**: they are your classmates' photographs and
for this course only.

Log in to Kaggle with the one account you will use for the whole semester,
open the link, press **Download**, and unpack the ZIP into this repository so
that `data/raw/train.csv` and `data/raw/train/` (1841 `.jpg` files) exist:

```bash
unzip ~/Downloads/archive.zip -d data/raw                                    # Linux, macOS
```

```powershell
Expand-Archive "$HOME\Downloads\archive.zip" -DestinationPath data\raw      # Windows PowerShell
```

The ZIP may be saved under another name; use whatever your browser saved.

**4. The caches and the tests.**

```bash
docker compose run --rm dev python -m src.data.dataset      # must report 1841 images; a few minutes
docker compose run --rm dev pytest                          # the two split tests fail: they are TASK 1
```

If the first command reports a number other than 1841, or cannot find
`train.csv`, the files are one folder too deep or too shallow:
`data/raw/train.csv` has to exist exactly there.

## Running the notebook from Docker

Nothing runs on your host except Docker. Jupyter runs inside the container
and you open it in a browser.

1. From the repository root, start Jupyter Lab:

   ```bash
   docker compose up lab
   ```

2. Wait for the line `Jupyter Server ... is running at` and open
   **http://localhost:8888** in your browser. There is no password or token.
3. In the file browser on the left, open `notebooks/lab3_baseline.ipynb`.
4. Edit the `src/` files in your own editor on the host. The project folder
   is mounted into the container, so the notebook sees every saved change;
   rerun the cell, and its `importlib.reload` picks the change up.
5. To stop, press `Ctrl+C` in the terminal, then run `docker compose down`.

**From VS Code instead of the browser.** Start the container as in step 1,
open the notebook in VS Code, click **Select Kernel → Existing Jupyter
Server**, and enter `http://localhost:8888`.

**If something goes wrong.**

| symptom | fix |
|---|---|
| `port is already allocated` | another Jupyter is running: `docker compose down`, or stop the other one |
| the notebook does not see your W&B key | you edited `.env` after starting: `docker compose down`, then `docker compose up lab` |
| `ModuleNotFoundError: src` | the notebook was opened from outside the container; use the browser at localhost:8888 |
| a cell keeps running old code | save the file, then rerun the cell; if it persists, *Kernel → Restart* |

Git runs on your host, not in the container: commit and push from a host
terminal in this folder.

---

## Everyday commands

```bash
docker compose run --rm dev bash                    # interactive shell
docker compose run --rm dev pytest                  # tests
docker compose run --rm dev ruff check .            # lint
docker compose run --rm dev python -m src.train --config configs/cnn.yaml
docker compose up lab                               # Jupyter Lab, localhost:8888
```

There is deliberately no Makefile. The commands above *are* the interface, and
when something breaks you will be debugging Docker rather than a wrapper around
it.

## How the container is put together

Three details, because they are the ones that break when you change something.

**The project is mounted, not copied.** Editing a file on your host changes it
in the container immediately. You only need `docker compose build` again when
`requirements.txt` or the `Dockerfile` changes.

**Torch comes from the base image and is deliberately not in
`requirements.txt`.** Installing it again would fetch a second, possibly
CPU-only build and quietly waste your GPU.

**The container runs as a normal user, not root**, so the files it writes into
your project are yours. If your host user id is not 1000 you will see permission
errors; rebuild with `docker compose build --build-arg UID=$(id -u)`.

---

## Layout

```
src/
  config.py            project root, target column, seed
  data/
    dataset.py         the table, the 768 px cache, the pixel table
    splits.py          the grouped split (TASK 1)
  features/
    handcrafted.py     classical image quality features
  models/
    baseline.py        features + a classical regressor
    mlp.py             the first network (TASK 2)
    cnn.py             convolutional models, a stub for later
  seed.py              reproducibility
  tracking.py          Weights & Biases
  train.py             the training script (TASKS 3-6 inside)
  evaluate.py          scoring and error breakdown
  predict.py           writes a submission file  <- read the contract in it

configs/               one YAML per experiment; lab3_mlp.yaml is the in-class run
scripts/check_env.py   verifies the container, prints versions and GPU status
notebooks/             exploration; move anything reusable into src/
tests/                 start with test_splits.py
data/raw/              what you were given, read only
data/work/             everything derived from it, regenerable
models/                checkpoints
submissions/           competition files
```

Nothing under `data/`, `models/` or `submissions/` is tracked by git.

## Making a Kaggle submission

One command produces one file:

```bash
docker compose run --rm dev python -m src.predict \
    --config configs/cnn.yaml \
    --checkpoint models/cnn_best.pt \
    --out submissions/cnn_best.csv
```

Then upload `submissions/cnn_best.csv` to the competition page.

The file has exactly two columns, `image_id` and `usability`, one row per test
image in the order of the competition's `sample_submission.csv`, every value a
float in [0, 1]. `src/predict.py` builds the output *from the sample
submission* rather than from a directory listing, so the ids and the row order
cannot drift, and it asserts the row count, uniqueness, no NaNs and the value
range before writing. Each of those has cost somebody a submission slot.

**One image in, one prediction out.** Your prediction for an image depends on
that image and your trained weights only, which is why the loop in `predict.py`
handles one image at a time and shares nothing between iterations. Batching for
speed is fine. Pooling anything across the test set is not; section 2 of the
project description says why.

**The checkpoint is part of the submission.** You hand in the weights that
produced your predictions, not a later or better version. Log every checkpoint
to W&B as an artifact of the run that made it and this question answers itself.

## The order of work

1. **Write the split.** Before any model. See below.
2. **Look at the data.** Plot the label distribution. Compare the best and
   worst photos *at full resolution*: a thumbnail hides the difference between
   a dark page and a blurred one.
3. **Train a network.** Pretrained backbone, regression head. Get one run
   working end to end, with a checkpoint and a submission file, before you
   improve anything.
4. **Compare it against the baseline.** The instructor publishes a classical
   baseline on the leaderboard, built from handcrafted image features, and your
   network has to beat it. Building your own version takes an afternoon and
   shows you what a whole-image statistic can and cannot see.
5. **Improve it.** Resolution first, then augmentation, architecture,
   ensembling. Change one thing per run.

## Write the split first

The same page appears in dozens of photographs, every layout exists in a
Hungarian and an English version typeset identically, and each photographer's
photos share a phone and a room. Split rows at random and near-identical photos
land in training and validation at the same time, and the score you report is
optimistic.

We measured it on this data with a ResNet-18 at 768 px, scoring the same model
four ways:

| validation scheme | RMSE |
|---|---|
| random 5-fold | 0.113 |
| folds by `group` | 0.136 |
| folds by `group_strict` | 0.143 |
| the held-out test set | **0.148** |

The random split is a fifth too good. `group`, in which no page and no
photographer appears twice, is still 8 % too good: the two language versions
of a layout sit in two neighbouring groups, and a network recognises the
layout. `group_strict` keeps both versions together, which is how the test set
was cut, and it lands on the test. The stronger the model, the more it profits
from every leak, so the mistake grows with your progress.

**Split by `group_strict`.** The training set has six strict units of ten pages
each, and each covers one or two document families, so which unit you hold out
matters: two of them are the only source of their family. The split therefore
has its own seed, separate from the model's, so that rerunning a model with
another seed does not move the validation photographs. `tests/test_splits.py`
asserts that no unit appears on both sides. It is a few lines, and it protects
every number you report this semester.

Implementing the random split too and measuring your own gap makes a far more
convincing paragraph in your report than a number we gave you.

## Things that will cost you if you get them wrong

**Augmentation that changes image quality.** Flips and small translations are
safe. Blur, brightness jitter and added noise change the very thing you are
predicting, so they corrupt the label. This is the most common way to make a
model quietly worse.

**Metadata as model input.** `page_id`, `family`, `font`, `size_pt`,
`photographer` and the rest exist only in `train.csv`; a test image is an image
and nothing else. Use the columns to split and to slice your error, never as
features.

**Measuring on `eval_ok = 0`.** Those 47 photographs were uploaded heavily
downscaled. Train on them if you like, but leave them out of every number you
report.

**Too little resolution.** At 384 px a ResNet-18 loses to the handcrafted
baseline; the defects that matter are a few pixels wide on a 2048 px frame.
Resize once into `data/work/` and start at 768.

**Reporting a single number.** RMSE tells you how well you did, not what to fix.
Break the error down by document family, language, font, font size and
photographer.

**Not fixing the seed.** You will compare dozens of runs. A difference smaller
than your run-to-run variance is not a result.

**Committing data or checkpoints.** `.gitignore` covers it, but check
`git status` before your first commit rather than after.

## Notes on the data

The photographs have their long side at 2048 px and no metadata. The fiducial
markers the labelling relied on have been painted out, so the four flat patches
near the page corners are not part of the document, and you cannot rectify the
page from them. The page edges are still there if you want to try.

The defects that matter most are the **local** ones: a smear across one column,
a crease through a paragraph. The same defect over text and over empty margin
looks nearly identical to a whole-image statistic and produces very different
labels. That is the core difficulty of this task, and it is why an average of
sharpness or brightness over the frame only gets you so far.

`train_annotations.csv` and `train_consensus.csv` hold human ratings of about a
third of the photographs, on a 1 to 5 scale, from several raters each. The
competition does not score them. Where people and the OCR engine disagree, and
whether the disagreement is structured, is an open question on data you helped
create.
