# do-crystal
# Population Balance Equation Solver for Crystallization Systems

A Python toolbox for solving Population Balance Equations (PBE) with applications to various crystallizer configurations. This repository integrates with the do-mpc (distributed optimization and model predictive control) framework to demonstrate PBE solutions for different crystallization processes.

## Features

- Robust PBE solver implementations
- Integration with do-mpc framework
- Example applications for common crystallizer configurations:
  - Mixed Suspension Mixed Product Removal (MSMPR)
  - Two-stage MSMPR
  - Batch crystallizer
  - Tubular crystallizer
- Visualization tools for crystal size distributions
- Performance optimized numerical methods

## Installation

```bash
git clone https://github.com/yourusername/pbe-crystallizer
cd pbe-crystallizer
pip install -r requirements.txt
```

## Dependencies

- do-mpc
- numpy
- scipy
- matplotlib
- casadi

## Usage

Each crystallizer example is contained in its own directory with a dedicated script and configuration files. To run an example:

```bash
python examples/msmpr/run_simulation.py
```

### Example Structure

```
examples/
├── msmpr/
├── two_stage_msmpr/
├── batch/
└── tubular/
```

## Examples

### 1. MSMPR Crystallizer

The Mixed Suspension Mixed Product Removal (MSMPR) example demonstrates:
- Steady-state crystal size distribution
- Effect of residence time on crystal growth
- Implementation of nucleation and growth kinetics

### 2. Two-Stage MSMPR

Extends the single-stage example to show:
- Coupled population balance equations
- Inter-stage mass transfer
- Optimization of stage conditions

### 3. Batch Crystallizer

Illustrates:
- Time-dependent crystal size evolution
- Batch process optimization
- Temperature profile effects

### 4. Tubular Crystallizer

Demonstrates:
- Spatial variation in crystal properties
- Plug flow assumptions
- Axial dispersion effects

## Documentation

Detailed documentation for each solver and example can be found in the `docs/` directory. This includes:
- Mathematical formulation of the PBE
- Numerical methods used
- Parameter selection guidelines
- Validation cases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Citation

If you use this toolbox in your research, please cite:

```bibtex
@software{pbe_crystallizer,
  author = {Your Name},
  title = {Population Balance Equation Solver for Crystallization Systems},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/pbe-crystallizer}
}
```

## Contact

- Your Name - [your.email@domain.com](mailto:your.email@domain.com)
- Project Link: [https://github.com/yourusername/pbe-crystallizer](https://github.com/yourusername/pbe-crystallizer)