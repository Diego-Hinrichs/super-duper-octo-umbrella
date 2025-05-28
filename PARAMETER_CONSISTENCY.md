# Parameter Consistency Across Implementations

This document describes the standardized command-line parameters across all simulation implementations, based on the BarnesHut GPU implementation as the reference.

## Standardized Parameters

All implementations now support the following consistent parameters:

### Core Parameters
- **`-n <value>`**: Number of bodies (default varies by implementation)
  - BarnesHut GPU: 10000
  - DirectSum GPU: 10000  
  - BarnesHut CPU: 1000
  - DirectSum CPU: 1000

- **`-s <value>`**: Number of iterations/simulation steps (default: 100)

- **`-l <value>`**: Domain size L for periodic boundary conditions
  - BarnesHut GPU: 2e+12
  - Others: 1e+06

### Distribution Parameters
- **`-dist <value>`**: Body distribution (galaxy, solar, uniform, random) (default: galaxy)
  - Note: BarnesHut CPU accepts this parameter for consistency but doesn't use it
  
- **`-mass <value>`**: Mass distribution (uniform, normal) (default: normal)

- **`-seed <value>`**: Random seed (default: 42)

### Space-Filling Curve Parameters
- **`-nosfc`**: Disable Space-Filling Curve ordering (flag)
- **`-curve <value>`**: SFC curve type (morton, hilbert) (default: morton)
- **`-freq <value>`**: Reordering frequency for fixed mode (default: 10)
  - Note: All implementations use dynamic reordering by default

### Implementation-Specific Parameters

#### GPU-Only Parameters
- **`-block <value>`**: CUDA block size (default: 256)
  - BarnesHut CPU and DirectSum CPU accept this for compatibility but don't use it

#### CPU-Only Parameters  
- **`-t <value>`**: Number of threads (default: 0 = auto)
  - 0 means use automatic thread detection

#### BarnesHut-Specific Parameters
- **`-theta <value>`**: Barnes-Hut opening angle parameter (default: 0.5)
  - Only available in BarnesHut implementations

#### BarnesHut GPU-Only Parameters
- **`-sm <value>`**: Shared memory size per block in bytes, 0=auto (default: 0)
- **`-leaf <value>`**: Maximum bodies per leaf node (default: 1)
- **`-energy`**: Calculate system energy (flag) - not currently used

## Usage Examples

### Basic Usage (same across all implementations)
```bash
# Run with 5000 bodies for 200 iterations
./main -n 5000 -s 200

# Use Hilbert curve with uniform mass distribution
./main -curve hilbert -mass uniform

# Disable SFC ordering
./main -nosfc

# Set domain size for periodic boundaries
./main -l 1e8
```

### GPU-Specific Usage
```bash
# Set CUDA block size
./main -block 512

# BarnesHut GPU with custom theta and leaf parameters
./main -theta 0.8 -leaf 2 -sm 1024
```

### CPU-Specific Usage
```bash
# Use 8 threads
./main -t 8

# Use single thread (sequential)
./main -t 1
```

## Parameter Validation

All implementations validate parameters and show help with:
```bash
./main --help
```

Note: The argument parser shows an error for `--help` but still displays the usage information.

## Implementation Notes

1. **Dynamic Reordering**: All implementations use dynamic reordering by default. The `freq` parameter is kept for compatibility but is not used in dynamic mode.

2. **Domain Size**: Different default values reflect the different scales used by each implementation, but all support the same parameter.

3. **Body Distribution**: While all implementations accept the `dist` parameter, some may not fully implement all distribution types.

4. **Thread Handling**: CPU implementations automatically detect the optimal number of threads when `-t 0` is used.

5. **Compatibility Parameters**: Some parameters (like `block` in CPU versions) are accepted for consistency but have no effect. 