# Adjoint ODE

Examples of gradients calculations using the adjoint state method with [OwlDE](https://github.com/owlbarn/owl_ode.git)

Currently, we have two examples: one solves an intial value problem and the other learns the parameters of a dynamical system as well as its initial condition.

You can run the two examples using the following commands:
```sh
dune exec examples/initial_condition_adjoint.exe
```

```sh
dune exec examples/params_adjoint.exe
```
