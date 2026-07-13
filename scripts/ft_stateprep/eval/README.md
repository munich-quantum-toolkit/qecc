# Fault-tolerant State Preparation

If you want to run evaluations on the example circuits in this directory, do the
following.

To estimate the logical error rate for a physical error rate `p_err`, run

```console
python estimate_logical_error_rate CODE -p p_err
```

The script prints 4 numbers: logical error rate per qubit, acceptance rate (if
using post-selection), number of logical errors, total number of shots

The python script has further options with which you can select which circuits
to construct, how many logical errors should occur before stopping and more.

To generate these values for a circuit for a range between `p_err = 0.5` and
`p_err = 0.00005`, the script `run_eval_on_code FILENAME ARGS` can be used. It
runs multiple instances of `estimate_logical_error_rate` in parallel (using GNU
Parallel) and stores the results in `FILENAME.csv`.
