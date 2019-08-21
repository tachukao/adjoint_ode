(* this is a simple application of OwlDE to learn
 * the initial condition of a dynamical system such 
 * that the dynamical system reaches the target at 
 * time t1 (i.e. x(t1) = target). We do so by running 
 * the adjoint dynamical system backwards in time to
 * compute the gradient with respect to the initial
 * condition. *)

open Owl

module Solver = Adjoint_ode.IVP.Make (struct
  let dim = 2
  let t0 = 0.
  let t1 = 10.
  let dt = 0.5

  (* differentiable dynamical equation dxdt = fd(x,t) *)
  let fd =
    let a = [| [| 0.; 1. |]; [| -1.; 0. |] |] |> Mat.of_arrays in
    let a = Algodiff.D.pack_arr a in
    fun x _t -> Algodiff.D.Maths.(a *@ x)
end)

(* learning rate and number of gradient iterations *)
let alpha = 1E-2
let max_iter = 1000

(* target final state *)
let target = Owl.Mat.of_array [| 1.; 0. |] 2 1 |> Algodiff.D.pack_arr

(* l2 loss at of the end point *)
let loss x1 = Algodiff.D.Maths.(l2norm_sqr' (x1 - target))

(* vanilla gradient descent *)
let rec learn =
  let grad = Solver.grad ~loss in
  fun step x0 l' ->
    (* calculate the loss l and its gradient with respect to x0 *)
    let dx0, l = grad x0 in
    (* calculate percentage change *)
    let pct_change = (l' -. l) /. l' in
    if step < max_iter && pct_change > 1E-4 && l > 1E-3
    then (
      if step mod 1 = 0 then Printf.printf "\riter %i | loss %4.3f | pct change %4.5f %!" step l pct_change;
      let x0 = Mat.(x0 - (alpha $* dx0)) in
      learn (succ step) x0 l)
    else x0


let () =
  (* learn initial condition *)
  let x0 = learn 0 Mat.(of_array [| 1.; 0. |] 2 1) 1E9 in
  let tspec =
    let t0 = Solver.t0
    and duration = Solver.duration in
    Owl_ode.Types.(T1 { t0; dt = 1E-2; duration })
  in
  let ts, xs = Owl_ode.Ode.odeint (module Owl_ode_odepack.Lsoda) Solver.f x0 tspec () in
  Mat.save_txt Mat.(transpose (ts @= xs)) "results/actual_xs";
  let x1 = Mat.col xs (-1) in
  Printf.printf "\n\nx1: actual and target \n%!";
  Mat.print x1;
  Mat.print (Algodiff.D.unpack_arr target)
