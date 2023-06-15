# Learning to Suggest Breaks: Sustainable Optimization of Long-Term User Engagement

Comments are welcome! 🦊🐇

## Environment setup
```
conda create -n lvml anaconda
conda activate lvml
conda install -c conda-forge scikit-surprise
```

## Repository structure
* `run_experiments.ipynb` - Run and analyze MovieLens/Goodreads experients. Results are written to `./output`.
* `analyze_experiment.ipynb` - Reproduce experiment analysis. Analysis of each dataset was performed on random seeds `[1,...,10]`.
* `schematic_diagrams.ipynb` - Reproduce schematic diagrams.
* `prepare_goodreads_dataset.ipynb` - Prepare the Goodreads dataset for analysis.
* `lvml/core/simulation.py` - Core simulation code, TPP simulator, ODE solver wrapper.
* `lvml/core/policy.py` - Policy optimizers (`LV`, `best-of`, `default`).
* `lvml/experiment/*` - Experiment-related utilities. 
