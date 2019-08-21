# Adjoint ODE

Examples of gradients calculations using the adjoint state method with [OwlDE](https://github.com/owlbarn/owl_ode.git)

Currently, we have two examples: one solves an intial value problem and the other learns the parameters of a dynamical system as well as its initial condition.

Example runs of the two examples, including the invocation commands can be found below:

```sh
$ mkdir results
$ dune exec examples/initial_value_problem.exe --profile=release
iter 203 | loss 0.001 | pct change 0.03960 

x1: actual and target 

           C0 
R0   0.970162 
R1 0.00882855 

   C0 
R0  1 
R1  0 
```

```sh
$ mkdir results
$ dune exec examples/params_adjoint.exe --profile=release
step 2520 | loss 0.01008 | pct change 0.00131
```
