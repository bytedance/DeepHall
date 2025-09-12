<p align="center">
  <img src=".github/img/deephall.svg" width="200">
</p>
<h1 align="center">DeepHall</h1>

Simulating the fractional quantum Hall effect (FQHE) with neural network variational Monte Carlo.

This repository contains the codebase for the paper [Describing Landau Level Mixing in Fractional Quantum Hall States with Deep Learning](https://doi.org/10.1103/PhysRevLett.134.176503). If you use this code in your work, please [cite our paper](CITATIONS.bib).

Currently, DeepHall supports running simulations with spin-polarized electrons on a sphere and has been tested with 1/3 and 2/5 fillings.

## Installation

DeepHall requires Python `>=3.11` and JAX `>=0.4.36`. It is highly recommended to install DeepHall in a separate virtual environment.

```bash
# Remember to activate your virtual environment
git clone https://github.com/bytedance/DeepHall
cd DeepHall
pip install -e . -r requirements.txt                  # Install CPU version
pip install -e ".[cuda12]" -r requirements.txt        # Download CUDA libraries from PyPI
pip install -e ".[cuda12_local]" -r requirements.txt  # Or, use local CUDA libraries
```


You can also use commands like `uv sync --extra cuda12` if you have [uv](https://docs.astral.sh/uv/) installed.

To further customize JAX installation, please refer to the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).

## Performing Simulations

### Command Line Invocation

You can use the `deephall train` command to run FQHE simulations. The configurations can be passed to DeepHall using the `key=value` syntax (see [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dot-list)). A simple example would be:

```bash
deephall train "system.nspins=[6,0]" system.flux=15 optim.iterations=100
```

In this example, we place six spin-polarized electrons on a sphere with a total magnetic flux of $2Q = 15$ passing through its surface. (Currently, only spin-polarized configurations are supported.) The radius of the sphere is implicitly set to $\sqrt{Q} = \sqrt{15/2}$. This configuration corresponds to a filling factor of $1/3$. Recall that on the sphere, the particle–flux relation is given by $2Q = N / \nu - \mathcal{S}$, where $\mathcal{S} = 3$ for $1/3$ filling. The reported energy includes contributions from both the kinetic term and the electron–electron interactions.

If you just want to test the installation, an even simpler example is the non-interacting case with a smaller network and batch size:

```bash
deephall train "system.nspins=[3,0]" system.flux=2 system.interaction_strength=0 optim.iterations=100 network.num_layers=2 batch_size=100
```

Details of available settings are available at [config.py](deephall/config.py).

### Python API

You can also use DeepHall from your Python script. For example:

```python
from deephall import Config, train

config = Config()
config.system.nspins = (3, 0)
config.system.flux = 2
config.system.interaction_strength = 0.0
config.optim.iterations = 100
config.network.num_layers = 2
config.batch_size = 100

train(config)
```

## Output

By default, the results directory is named like `DeepHall_n3l2_xxxxxx_xxxxxx`. You can configure the output location with the `log.save_path` config, which can be any writable path on the local machine or a remote path supported by [universal_pathlib](https://github.com/fsspec/universal_pathlib).

In the results directory, the file you will need most of the time is `train_stats.csv`, which contains the energy, angular momentum, and other useful quantities per step. The checkpoint files like `ckpt_000099.npz` store Monte Carlo walkers and neural network parameters so that the wavefunction can be analyzed, and the training can be resumed.

## Wavefunction Analysis with NetObs

DeepHall contains a `netobs_bridge` module to calculate the pair correlation function, overlap with the Laughlin wavefunction, and the one-body reduced density matrix. With [NetObs](https://github.com/bytedance/netobs) installed:

```bash
# Energy
netobs deephall unused energy --with steps=2000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/energy
# Overlap
netobs deephall unused deephall@overlap --with steps=50 --net-restore save_path/ckpt_000099.npz --ckpt save_path/overlap
# Pair correlation function
netobs deephall unused deephall@pair_corr --with steps=100000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/pair_corr
# 1-RDM
netobs deephall unused deephall@one_rdm --with steps=20000 --net-restore save_path/ckpt_000099.npz --ckpt save_path/1rdm
```

## Adding a New Neural Network Wavefunction

To implement a custom neural network wavefunction, follow the steps below:

### Step 1: Implement the Network

Create a new file in the `deephall/networks/` directory, for example, `deephall/networks/mynet.py`. You can use the existing implementation in `deephall/networks/mhpo.py` as a reference. Below is a minimal example of how to structure your network:

```python
from flax import linen as nn
from deephall.config import System

# This line is essential as it registers the module with DeepHall
__all__ = ["MyNet"]


class MyNet(nn.Module):
    """A custom neural network wavefunction."""
    system: System  # System configuration
    num_layers: int = 2  # Number of layers (default: 2)

    @nn.compact
    def __call__(self, electrons):
        """
        Forward pass of the network.

        Args:
            electrons: Input electron coordinates.

        Returns:
            logpsi: The log wavefunction value.
        """
        ...
        return logpsi
```

### Step 2: Integrate the Network into DeepHall

#### Using the Network via CLI

To use `MyNet` from the command line, specify the network type and its parameters as follows:

```bash
deephall train "system.nspins=[3,0]" system.flux=2 network.type=mynet network.num_layers=1
```

#### Using the Network via API

To integrate `MyNet` programmatically, configure the settings in Python as shown below:

```python
from deephall import Config, train

config = Config()
config.system.nspins = (3, 0)
config.system.flux = 2
config.network.type = "mynet"
config.network.num_layers = 2

train(config)
```

### Additional Notes

- **Custom File Locations**: While placing your network file in `deephall/networks/` is recommended, you can store it elsewhere. In such cases, specify the full path to the file in the `network.type` parameter. For example:

  ```bash
  deephall train "system.nspins=[3,0]" system.flux=2 network.type=/path/to/mynet.py network.num_layers=1
  ```

- **Best Practices**: Ensure that your network class inherits from `nn.Module` and adheres to the expected interface. The `__all__` variable is crucial for module discovery by DeepHall.

## Citing Our Paper

If you use this code in your work, please cite the following paper:

```bib
@article{PhysRevLett.134.176503,
  title = {Describing {{Landau}} Level Mixing in Fractional Quantum {{Hall}} States with Deep Learning},
  author = {Qian, Yubing and Zhao, Tongzhou and Zhang, Jianxiao and Xiang, Tao and Li, Xiang and Chen, Ji},
  journal = {Phys. Rev. Lett.},
  volume = {134},
  issue = {17},
  pages = {176503},
  numpages = {8},
  year = {2025},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.134.176503},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.134.176503}
}
```
