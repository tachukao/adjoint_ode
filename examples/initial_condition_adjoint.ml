(* this is a simple application of OwlDE to learn
 * the initial condition of a dynamical system such 
 * that the dynamical system reaches the target at 
 * time t1 (i.e. x(t1) = target). We do so by running 
 * the adjoint dynamical system backwards in time to
 * compute the gradient with respect to the initial
 * condition. *)

open Owl
open Owl_ode
open Owl_ode_sundials

(* dimension of state-space *)
let n = 2

(* helper function *)
let g =
  let a = [| [| 0.; 1. |]; [| -1.; 0. |] |] |> Mat.of_arrays in
  let a = Algodiff.D.pack_arr a in
  fun x -> Algodiff.D.Maths.(a *@ x)


(* define the dynamical system equation dx/dt = f(x,t) *)
let f x _t =
  let x = Algodiff.D.pack_arr x in
  let dxdt = g x in
  Algodiff.D.unpack_arr dxdt


(* define the corresponding adjoint dynamical system 
 * ds/dt = adj_f(s,t). The adjoint state s = [ x @= a ]
 * is a vertical concatenation of x and a, where a = dl/dx 
 * is the derivative of the loss l with respect to the state
 * x *)

let adj_f =
  (* function to extract state and adjoint from adjoint state *)
  let extract s =
    let x = Mat.(s.${[ [ 0; pred n ] ]}) in
    let a = Mat.(s.${[ [ n; -1 ] ]}) in
    x, a
  in
  fun s _t ->
    let x, a = extract s in
    let x = Algodiff.D.pack_arr x in
    let a = Algodiff.D.pack_arr a in
    let dx = Algodiff.D.Maths.neg (g x) in
    let da = Algodiff.D.(Maths.neg (jacobianTv g x a)) in
    let dx = Algodiff.D.unpack_arr dx in
    let da = Algodiff.D.unpack_arr da in
    Owl.Mat.(dx @= da)


(* learning rate and number of gradient iterations *)
let alpha = 1E-2
let max_iter = 1000

(* target final state *)
let target = Owl.Mat.of_array [| 1.; 0. |] 2 1 |> Algodiff.D.pack_arr

(* l2 loss at of the end point *)
let loss x1 = Algodiff.D.Maths.(l2norm_sqr' (x1 - target))

(* define gradient with respect to end state x1 *)
let dloss = Algodiff.D.grad loss

(* time specification *)
let t0 = 0.
let t1 = 10.
let duration = t1 -. t0
let dt = duration
let tspec = Types.(T1 { t0; dt; duration })

(* forward pass through time *)
let forward x0 =
  let _, xs = Ode.odeint (module Owl_Cvode) f x0 tspec () in
  Mat.col xs (-1)


(* backward pass through time *)
let backward x1 =
  let c = loss (Algodiff.D.pack_arr x1) |> Algodiff.D.unpack_flt in
  let s1 =
    let a1 = dloss Algodiff.D.(pack_arr x1) |> Algodiff.D.unpack_arr in
    Mat.(x1 @= a1)
  in
  let tspec = Types.(T1 { t0 = t1; dt = -.dt; duration = -.duration }) in
  let _, adj_ss = Ode.odeint (module Owl_Cvode) adj_f s1 tspec () in
  c, Mat.col adj_ss (-1)


(* vanilla gradient descent *)
let rec learn step x0 l' =
  (* run the dynamics forward from current guess of x0 *)
  let x1 = forward x0 in
  (* calculate the loss l and its gradient with respect to x0 *)
  let l, dx0 =
    let l, x0 = backward x1 in
    l, Mat.(x0.${[ [ n; -1 ] ]})
  in
  let pct_change = (l' -. l) /. l in
  if step < max_iter && pct_change > 1E-4
  then (
    if step mod 1 = 0 then Printf.printf "\riter %i | loss %3.3f %!" step l;
    let x0 = Mat.(x0 - (alpha $* dx0)) in
    learn (succ step) x0 l)
  else x0


let () =
  (* initial guess *)
  let x0 = Mat.of_array [| 1.; 0. |] 2 1 in
  (* learn true x0 *)
  let x0 = learn 0 x0 1E9 in
  let tspec = Types.(T1 { t0; dt = 1E-2; duration }) in
  let ts, xs = Ode.odeint (module Owl_Cvode) f x0 tspec () in
  Mat.save_txt Mat.(transpose (ts @= xs)) "actual_xs";
  let x1 = Mat.col xs (-1) in
  Printf.printf "\n\nx1: actual and target \n%!";
  Mat.print x1;
  Mat.print (Algodiff.D.unpack_arr target)
