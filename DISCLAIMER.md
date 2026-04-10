# Disclaimer

This software is a research prototype developed for academic investigation of
Model Predictive Control algorithms for closed-loop deep brain stimulation (DBS).

## Not for Clinical Use

This software is **not** approved for clinical use, medical device deployment,
or use in any safety-critical application. It has not been evaluated for
regulatory compliance (FDA, CE, or equivalent).

## Synthetic Demo Data

The weight files (`generated/network_weights.hpp`, `generated/koopman_weights.hpp`)
and reference test case files (`firmware/reference_test_case.hpp`,
`firmware/koopman_reference_test_case.hpp`) contain **synthetic, randomly
generated values** for demonstration purposes. They do not represent any trained
model or patient data.

To use this solver with a real model, you must supply your own trained weights
in the format described in `include/stable_neuron_solver/weights.hpp`.

## No Warranty

This software is provided "as is" without warranty of any kind. See the
LICENSE file for full terms.
