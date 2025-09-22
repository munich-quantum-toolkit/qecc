# Analog Information Decoding

This submodule provides means to conduct decoding simulations for quantum CSS codes with
analog syndrome information as proposed in the corresponding paper {cite:labelpar}`berent2023analog`.
Proper integration and setup is a work in progress.
The main functionality is provided in the {code}`simulators` module, which contains the main classes
that can be used to conduct several types of simulations:

- {code}`ATD_Simulator`: Analog Tanner graph decoding,
- {code}`Single-Shot Simulator`: Analog Single-Shot decoding with meta checks.
- {code}`QSS_Simulator`: Quasi-Single-Shot decoding, and

Moreover, {code}`memory_experiment` contains methods for analog overlapping window decoding, in
particular the {code}`decode_multiround` method.

## Results

The {code}`results` directory contains the results used in the paper {cite:labelpar}`berent2023analog`.

## Codes

The {code}`codes` directory contains the parity-check matrices of the codes used in the paper {cite:labelpar}`berent2023analog`.
Three dimensional toric codes can either be constructed with the hypergraph product construction
or with a library, e.g., panqec {cite:labelpar}`huang2023panceq`.

## Code construction

The {code}`code_construction` directory contains the code used to construct higher-dimensional hypergraph
product codes and used the {code}`compute_distances.sh` script to automatically compute bounds on the
distances of the constructed codes using the GAP library QDistRnd {cite:labelpar}`pryadko2023qdistrnd`.

## Utils

Here we present an overview of the functionality of the utils package.

### Plotting

The {code}`plotting` directory contains the code used to plot the results in the paper {cite:labelpar}`berent2023analog`.

### Data Utils

We have implemented several utility functions as part of this package that might be of independent
interest for decoding simulations and data analysis. These are provided in the {code}`data_utils` module.

### Simulation Utils

This module contains functionality needed throughout different simulation types.

### Dependencies

The used BP+OSD implementation, as well as our implementation of the soft-syndrome minimum-sum decoder are provided
in the ldpc2 package a preliminary beta version of which is available on github {cite:labelpar}`roffe2023ldpc`.
