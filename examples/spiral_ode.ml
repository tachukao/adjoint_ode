(* neural ode for learning a spiral *)
open Owl
open Owl_ode
open Adjoint_ode.Solvers

(* dimension of state-space *)
let n = 8
let p = 50
let n_obs = 2

(* time specification *)
let t0 = 0.
let t1 = Const.pi *. 3.
let duration = t1 -. t0
let dt = duration
let tspec = Types.(T1 { t0; dt; duration })

(* define the dynamical system model: 
   dx/dt = f(x,t) = w2 *@ (tanh (w1 *@ x + b)), where w1, w2, b are 
   paramters that we will also learn *)
let g w1 b w2 x = Algodiff.D.Maths.(w2 *@ tanh ((w1 *@ x) + b))
let f w1 b w2 x _t = g w1 b w2 x

(* the adjoint state space is a concatenation of the current state x
   derivative of the loss with respective to x and the derivation
   of the loss with respect to parameters w *)
let extract s =
  let open Algodiff.D in
  let x = Maths.get_slice [ [ 0; pred n ] ] s in
  let dldx = Maths.get_slice [ [ n; pred (2 * n) ] ] s in
  let dldw1 = Maths.get_slice [ [ 2 * n; (2 * n) + (n * p) - 1 ] ] s in
  let dldb = Maths.get_slice [ [ (2 * n) + (n * p); pred (2 * n) + ((n + 1) * p) ] ] s in
  let dldw2 = Maths.get_slice [ [ (2 * n) + ((n + 1) * p); -1 ] ] s in
  x, dldx, dldw1, dldb, dldw2


(* define the negative adjoint dynamical system: 
   ds/dt = negadjf(s,w,t) = 
   [-dx/dt; -d(dldx)/dt; -d(dl/dw1)/dt; -d(dl/db)/dt; -d(dl/dw2)/dt] 
   for integrating the adjoint state backwards in time *)
let negadjf dloss w1 b w2 s t =
  let open Algodiff.D in
  let x, dldx, _, _, _ = extract s in
  let w1 = primal w1 in
  let b = primal b in
  let w2 = primal w2 in
  let dx = g w1 b w2 x |> primal in
  let g1 x = g w1 b w2 x in
  let g2 w1 = g w1 b w2 x in
  let g3 b = g w1 b w2 x in
  let g4 w2 = g w1 b w2 x in
  (* note: we use dloss (t1 -. t) instead of dloss t because we are integrating
     backwards in time *)
  let ddldx = Maths.neg Maths.(jacobianTv g1 x dldx + dloss (t1 -. t) x) |> primal
  in
  let ddldw1 =
    Maths.neg (jacobianTv g2 w1 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  let ddldb =
    Maths.neg (jacobianTv g3 b dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  let ddldw2 =
    Maths.neg (jacobianTv g4 w2 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  Maths.concatenate ~axis:0 [| dx; ddldx; ddldw1; ddldb; ddldw2 |] |> Maths.neg


(* learnig rate and maximum gradient iterations *)
let alpha = 1E-3
let max_iter = 100000

(* target at time t *)
let target t =
  let r = 3. /. (t1 +. 1. -. t) in
  [| (r *. sin t) -. 1.; r *. cos t |]
  |> fun x -> Owl.Mat.of_array x n_obs 1 |> Algodiff.D.pack_arr


(* loss with respect to state x at time t *)
let loss t =
  let target = target t in
  fun x -> Algodiff.D.Maths.(l2norm_sqr' (get_slice [ [ 0; pred n_obs ] ] x - target))


(* derivative of loss with repsect to x at time t *)
let dloss t = Algodiff.D.grad (loss t)
let negadjf = negadjf dloss

(* running the dynamics forward in time *)
let forward loss w1 b w2 x0 =
  let s0 = Algodiff.D.Maths.concatenate ~axis:0 [| x0; Algodiff.D.Mat.zeros 1 1 |] in
  let f s t =
    let x = Algodiff.D.Maths.get_slice [ [ 0; pred n ] ] s in
    let dx = f w1 b w2 x t in
    let dl = loss t x in
    Algodiff.D.Maths.concatenate ~axis:0 [| dx; Algodiff.D.(Maths.(Mat.ones 1 1 * dl)) |]
  in
  let _, ss = Ode.odeint (module CSolver) f s0 tspec () in
  let s0 = Mat.col ss (-1) in
  Mat.get_slice [ [ 0; pred n ] ] s0 |> Algodiff.D.pack_arr, Mat.get s0 n 0


(* running the dynamics backward in time and calculating gradients *)
let backward w1 b w2 x1 =
  (* times at which to evaluate loss and gradients *)
  let negadjf = negadjf w1 b w2 in
  let open Algodiff.D in
  let s1 =
    let dldx = dloss t1 x1 |> primal in
    Maths.concatenate
      ~axis:0
      [| x1; dldx; Mat.(zeros (p * n) 1); Mat.(zeros p 1); Mat.(zeros (n * p) 1) |]
  in
  let _, adjs = Ode.odeint (module CSolver) negadjf s1 tspec () in
  let s0 = Owl.Mat.col adjs (-1) |> pack_arr in
  let x0, dldx, dldw1, dldb, dldw2 = extract s0 in
  let dldw1 = Maths.reshape dldw1 [| p; n |] in
  let dldb = Maths.reshape dldb [| p; 1 |] in
  let dldw2 = Maths.reshape dldw2 [| n; p |] in
  x0, dldx, dldw1, dldb, dldw2


(* helper function to save xs *)
let save_xs x0 w1 b w2 =
  let tspec = Types.(T1 { t0; dt = 1E-2; duration }) in
  let ts, xs = Ode.odeint (module CSolver) (f w1 b w2) x0 tspec () in
  Owl.Mat.save_txt Owl.Mat.(transpose (ts @= xs)) "actual_xs";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr w1) "w1";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr w2) "w2";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr b) "b"


(* gradient + RMSprop to speed up learning *)
let rec learn step x0 w1 b w2 dldw12 dldb2 dldw22 l' =
  let x1, l = forward loss w1 b w2 x0 in
  let _, _, dldw1, dldb, dldw2 = backward w1 b w2 x1 in
  let pct_change = (l' -. l) /. l' in
  if step < max_iter && l > 1E-3
  then (
    let open Algodiff.D in
    (* Adjoint_ode.Helper.print_algodiff_dim w1;
    Adjoint_ode.Helper.print_algodiff_dim dldw1; *)
    let dldw12 = Maths.((F 0.1 * sqr dldw1) + (F 0.9 * dldw12)) in
    let dldb2 = Maths.((F 0.1 * sqr dldb) + (F 0.9 * dldb2)) in
    let dldw22 = Maths.((F 0.1 * sqr dldw2) + (F 0.9 * dldw22)) in
    let w1 = Maths.(w1 - (F alpha * dldw1 / sqrt dldw12)) in
    let b = Maths.(b - (F alpha * dldb / sqrt dldb2)) in
    let w2 = Maths.(w2 - (F alpha * dldw2 / sqrt dldw22)) in
    (* let x0 = Maths.(x0 - (F alpha * dldx0 / sqrt dldx02)) in *)
    if step mod 10 = 0
    then Printf.printf "\rstep %i | loss %4.5f | pct change %4.7f %!" step l pct_change;
    (* save the dynamics every 100 gradient steps *)
    if step mod 100 = 0 then save_xs x0 w1 b w2;
    learn (succ step) x0 w1 b w2 dldw12 dldb2 dldw22 l)
  else x0, w1, b, w2


let () =
  (* save target *)
  let target_xs =
    let ts = Mat.linspace t0 t1 (succ (int_of_float (duration /. 1E-2))) in
    ts
    |> Mat.to_array
    |> Array.map (fun t -> target t |> Algodiff.D.unpack_arr)
    |> Mat.concatenate ~axis:1
    |> fun x -> Mat.(ts @= x) |> Mat.transpose
  in
  Mat.save_txt target_xs "target_xs";
  (* initial guess of inital condition *)
  let x0 = Algodiff.D.Maths.concatenate ~axis:0 [|target 0.; Algodiff.D.Mat.gaussian (n-n_obs) 1|] in
  (* initial guess of paramter w *)
  let w1 = Algodiff.D.Mat.uniform ~a:(-0.01) ~b:0.01 p n in
  let b = Algodiff.D.Mat.uniform ~a:(-0.01) ~b:0.01 p 1 in
  let w2 = Algodiff.D.Mat.uniform ~a:(-0.01) ~b:0.01 n p in
  (* learning x0 and w *)
  let x0, w1, b, w2 =
    learn
      0
      x0
      w1
      b
      w2
      Algodiff.D.Mat.(ones p n)
      Algodiff.D.Mat.(ones p 1)
      Algodiff.D.Mat.(ones n p)
      1E9
  in
  (* save learnt dynamics *)
  save_xs x0 w1 b w2
