# Usage guide

Run terminal commands from the project folder containing `README.md`, `src/`,
and `docker-compose.yml`. Keep the bundled workbook at
`data/Real estate valuation data set.xlsx`.

## 1. Choose how to run

### Without Docker

With Python and Jupyter already installed, install the project dependencies in
the Python environment you will use, then start Jupyter:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Use `python3` instead of `python` if that is your interpreter command. Open the
URL printed in the terminal and create a Python notebook in the **project root**.
Use a kernel from the same environment where you installed the dependencies.
For the classic interface, use `python -m jupyter notebook` instead.

### With Docker

Install/start Docker with Docker Compose available. Host Python and Jupyter are
not used by this method. Build and start the project:

```bash
docker compose up --build
```

Open the localhost URL printed in the logs, including its token (port `8888`).
In Jupyter, open `/home/jovyan/work` and create a Python notebook there; it should
be alongside `src/` and `data/`. Dependencies are installed by the image build.

For later starts, use `docker compose up`. Rebuild after changing dependencies.
Stop with `Ctrl+C`, or run `docker compose down` from another terminal in the
project folder. The project is mounted into the container, so generated results
are also saved on your computer.

## 2. Run an analysis in Jupyter

Run any command below in a notebook code cell with **Shift+Enter**. Each script
runs independently; regression scripts do not require running PCA analysis first.

```python
# Exploratory PCA: variance, scores, components, and scree chart
%run src/pca_analysis.py

# Linear regression: original features versus PCA
%run src/linear_regression.py --pca-components 3 --cv-folds 5

# k-NN: select k independently for original features and PCA
%run src/knn_regression.py --pca-components 3 --cv-folds 5 --k-values 3 5 7 9 11 15 20

# Decision trees: select depth and minimum leaf size independently
%run src/decision_tree.py --pca-components 3 --cv-folds 5 --max-depth-values 2 3 4 5 none --min-samples-leaf-values 1 5 10 20 --random-state 42
```

Tables print below the cell; graphs display inline and are saved as PNG and SVG.
In tree arguments, `none` means unlimited maximum depth.

## 3. Run from a terminal instead

Without Docker, replace `%run` in any analysis command with `python`:

```bash
python src/linear_regression.py --pca-components 3 --cv-folds 5
```

With the Docker service running, use `docker compose exec`:

```bash
docker compose exec -e MPLBACKEND=Agg jupyter python src/linear_regression.py --pca-components 3 --cv-folds 5
```

Substitute any of the other scripts and its arguments from section 2.
`MPLBACKEND=Agg` saves graphs without opening desktop windows. In a local terminal,
a plot window may pause execution until you close it; Jupyter displays inline.

## 4. Find and interpret results

| Analysis | Output folder | Main outputs |
|---|---|---|
| PCA | `results/pca/` | `summary.csv`, component/score/variance CSVs, `scree.png` / `.svg` |
| Linear regression | `results/regression/` | `metrics.csv`, CV scores, predictions, diagnostic plots |
| k-NN | `results/knn/` | `results.csv`, CV scores for each k, predictions, CV curve and diagnostic plots |
| Decision trees | `results/decision_tree/` | `summary.csv`, CV grid/fold scores, predictions/residuals, CV curves, heatmaps, tree diagrams and diagnostic plots |

Regression folders include `cv_metrics.csv` (individual folds), `cv_summary.csv`
(means and sample SDs), and `predictions.csv`. `regression_diagnostics.png` / `.svg`
compares actual-versus-predicted and residual-versus-predicted test values for
both representations. Lower MAE/RMSE and higher R² indicate better performance.

Defaults preserve **331 development / 83 final-test rows**, with split seed 42.
Five-fold CV uses development rows only; preprocessing is fitted inside each fold.
k-NN and trees select the lowest mean validation RMSE among the supplied candidates,
then refit on all development rows before final-test evaluation. Exploratory PCA
uses the full workbook independently.

Rerunning an analysis overwrites its files in the chosen output folder. Add
`--output-dir results/my_run` to keep a separate run. All scripts accept
`--input "path/to/workbook.xlsx"` for another workbook with the same required
columns, and `--help` to list arguments. Keep the default split settings when
comparing models. The README describes each pipeline and output in more detail.

## 5. Verify or troubleshoot

Run all tests locally or in the running container:

```bash
python -m unittest discover -s tests -v
docker compose exec -e MPLBACKEND=Agg jupyter python -m unittest discover -s tests -v
```

- **Missing Python package:** check the selected kernel. From a notebook in the
  project root, run `%pip install -r requirements.txt`, then restart the kernel.
- **Script not found:** use `%pwd` in Jupyter to check the folder; use
  `%cd /path/to/project` locally or `%cd /home/jovyan/work` in Docker.
- **Port 8888 occupied:** stop the other service, or change the Compose port mapping
  to `8889:8888` and open `http://localhost:8889` using the token from the logs.
- **No inline chart:** use `%run`, rather than `!python`, in notebook cells;
  saved images remain available in the output folder.
