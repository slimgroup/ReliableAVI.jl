<h1 align="center">Reliable amortized variational inference with physics-based latent distribution correction</h1>

Code to partially reproduce results in "Reliable amortized variational
inference with physics-based latent distribution correction".


## Installation

Before starting installing the required packages in Julia, make sure
you have `matplotlib` and `seaborn` installed in your Python environment 
since we depend on `PyPlot.jl` and `Seaborn.jl` for creating figures.

Run the command below to install the required packages.

```bash
julia -e 'Pkg.add("DrWatson.jl")'
julia --project -e 'using Pkg; Pkg.instantiate()'
```


The training dataset will download automatically into
`data/training-data/` directory upon running your first example.

## Questions

Please contact alisk@rice.edu for questions.

## Author

Ali Siahkoohi


