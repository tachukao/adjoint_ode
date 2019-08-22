(* neural ode for learning a spiral *)
open Owl
open Owl_ode
open Adjoint_ode.Solvers

(* dimension of state-space *)
let n = 4
let p = 100
let n_obs = 2

(* time specification *)
let t0 = 0.
let t1 = Const.pi *. 6.
let duration = t1 -. t0
let dt = duration
let tspec = Types.(T1 { t0; dt; duration })

(* define the dynamical system model: 
   dx/dt = f(x,t) = tanh (w2 *@ (tanh (w1 *@ x + b1)) + b2), 
   where w1, w2, b1, b2 are paramters that we will also learn *)
let g w1 b1 w2 b2 x =
  let open Algodiff.D.Maths in
  let a1 = tanh ((w1 *@ x) + b1) in
  tanh ((w2 *@ a1) + b2)


let f w1 b1 w2 b2 x _t = g w1 b1 w2 b2 x

(* the adjoint state space is a concatenation of the current state x
   derivative of the loss with respective to x and the derivation
   of the loss with respect to parameters w *)
let extract s =
  let open Algodiff.D in
  let x = Maths.get_slice [ [ 0; pred n ] ] s in
  let dldx = Maths.get_slice [ [ n; pred (2 * n) ] ] s in
  let dldw1 = Maths.get_slice [ [ 2 * n; pred (2 * n) + (n * p) ] ] s in
  let dldb1 =
    Maths.get_slice [ [ (2 * n) + (n * p); pred (2 * n) + ((n + 1) * p) ] ] s
  in
  let dldw2 =
    Maths.get_slice [ [ (2 * n) + ((n + 1) * p); pred (2 * n) + (((2 * n) + 1) * p) ] ] s
  in
  let dldb2 = Maths.get_slice [ [ (2 * n) + (((2 * n) + 1) * p); -1 ] ] s in
  x, dldx, dldw1, dldb1, dldw2, dldb2


(* define the negative adjoint dynamical system: 
   ds/dt = negadjf(s,w,t) = 
   [-dx/dt; -d(dldx)/dt; -d(dl/dw1)/dt; -d(dl/db)/dt; -d(dl/dw2)/dt] 
   for integrating the adjoint state backwards in time *)
let negadjf dloss w1 b1 w2 b2 s t =
  let open Algodiff.D in
  let x, dldx, _, _, _, _ = extract s in
  let w1 = primal w1 in
  let b1 = primal b1 in
  let b2 = primal b2 in
  let w2 = primal w2 in
  let dx = g w1 b1 w2 b2 x |> primal in
  let g1 x = g w1 b1 w2 b2 x in
  let g2 w1 = g w1 b1 w2 b2 x in
  let g3 b1 = g w1 b1 w2 b2 x in
  let g4 w2 = g w1 b1 w2 b2 x in
  let g5 b2 = g w1 b1 w2 b2 x in
  (* note: we use dloss (t1 -. t) instead of dloss t 
     because we are integrating backwards in time *)
  let ddldx = Maths.neg Maths.(jacobianTv g1 x dldx + dloss (t1 -. t) x) |> primal in
  let ddldw1 =
    Maths.neg (jacobianTv g2 w1 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  let ddldb1 =
    Maths.neg (jacobianTv g3 b1 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  let ddldw2 =
    Maths.neg (jacobianTv g4 w2 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  let ddldb2 =
    Maths.neg (jacobianTv g5 b2 dldx) |> fun x -> Maths.reshape x [| -1; 1 |] |> primal
  in
  Maths.concatenate ~axis:0 [| dx; ddldx; ddldw1; ddldb1; ddldw2; ddldb2 |] |> Maths.neg


(* target at time t *)
let target t =
  let r = 1. /. (t1 +. 1. -. t) in
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
let forward loss w1 b1 w2 b2 x0 =
  let s0 = Algodiff.D.Maths.concatenate ~axis:0 [| x0; Algodiff.D.Mat.zeros 1 1 |] in
  let f s t =
    let x = Algodiff.D.Maths.get_slice [ [ 0; pred n ] ] s in
    let dx = f w1 b1 w2 b2 x t in
    let dl = loss t x in
    Algodiff.D.Maths.concatenate ~axis:0 [| dx; Algodiff.D.(Maths.(Mat.ones 1 1 * dl)) |]
  in
  let _, ss = Ode.odeint (module CSolver) f s0 tspec () in
  let s0 = Mat.col ss (-1) in
  Mat.get_slice [ [ 0; pred n ] ] s0 |> Algodiff.D.pack_arr, Mat.get s0 n 0


(* running the dynamics backward in time and calculating gradients *)
let backward w1 b1 w2 b2 x1 =
  (* times at which to evaluate loss and gradients *)
  let negadjf = negadjf w1 b1 w2 b2 in
  let open Algodiff.D in
  let s1 =
    Maths.concatenate
      ~axis:0
      [| x1
       ; Mat.(zeros n 1)
       ; Mat.(zeros (p * n) 1)
       ; Mat.(zeros p 1)
       ; Mat.(zeros (n * p) 1)
       ; Mat.(zeros n 1)
      |]
  in
  let _, adjs = Ode.odeint (module CSolver) negadjf s1 tspec () in
  let s0 = Owl.Mat.col adjs (-1) |> pack_arr in
  let x0, dldx, dldw1, dldb1, dldw2, dldb2 = extract s0 in
  let dldw1 = Maths.reshape dldw1 [| p; n |] in
  let dldb1 = Maths.reshape dldb1 [| p; 1 |] in
  let dldw2 = Maths.reshape dldw2 [| n; p |] in
  let dldb2 = Maths.reshape dldb2 [| n; 1 |] in
  x0, dldx, dldw1, dldb1, dldw2, dldb2


(* helper function to save xs *)
let save x0 w1 b1 w2 b2 =
  let tspec = Types.(T1 { t0; dt = 1E-2; duration }) in
  let ts, xs = Ode.odeint (module CSolver) (f w1 b w2) x0 tspec () in
  (try Unix.mkdir "results" 0o777 with
  | Unix.Unix_error (Unix.EEXIST, _, _) -> ());
  Owl.Mat.save_txt Owl.Mat.(transpose (ts @= xs)) "results/actual_s";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr w1) "results/w1";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr w2) "results/w2";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr b1) "results/b1";
  Owl.Mat.save_txt Algodiff.D.(unpack_arr b2) "results/b2"


(* gradient + RMSprop to speed up learning 
   TODO: use LBFGS *)
let rec learn step x0 w1 b1 w2 b2 vw1 vb1 vw2 vb2 l' lr =
  let x1, l = forward loss w1 b1 w2 b2 x0 in
  let _, _, dldw1, dldb1, dldw2, dldb2 = backward w1 b1 w2 b2 x1 in
  let pct_change = (l' -. l) /. l' in
  if step < 100000 && l > 1E-3
  then (
    let open Algodiff.D in
    (* Adjoint_ode.Helper.print_algodiff_dim w1;
    Adjoint_ode.Helper.print_algodiff_dim dldw1; *)
    let lr = if pct_change < 0. then Maths.(lr / (F 2.)) else Maths.(lr * (F 1.01)) in
    let vw1 = Maths.((F 0.1 * sqr dldw1) + (F 0.9 * vw1)) in
    let vb1 = Maths.((F 0.1 * sqr dldb1) + (F 0.9 * vb1)) in
    let vw2 = Maths.((F 0.1 * sqr dldw2) + (F 0.9 * vw2)) in
    let vb2 = Maths.((F 0.1 * sqr dldb2) + (F 0.9 * vb2)) in
    let w1 = Maths.(w1 - (lr * dldw1 / (sqrt vw1 + F 1E-9))) in
    let b1 = Maths.(b1 - (lr * dldb1 / (sqrt vb1 + F 1E-9))) in
    let w2 = Maths.(w2 - (lr * dldw2 / (sqrt vw2 + F 1E-9))) in
    let b2 = Maths.(b2 - (lr * dldb2 / (sqrt vb2 + F 1E-9))) in
    (* let x0 = Maths.(x0 - (F lr * dldx0 / sqrt dldx02)) in *)
    if step mod 10 = 0
    then Printf.printf "\rstep %i | loss %4.5f | pct change %4.7f | lr %4.5f %!" step l pct_change (unpack_flt lr);
    (* save the dynamics every 100 gradient steps *)
    if step mod 100 = 0 then save x0 w1 b1 w2 b2;
    learn (succ step) x0 w1 b1 w2 b2 vw1 vb1 vw2 vb2 l lr)
  else x0, w1, b1, w2, b2


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
  Mat.save_txt target_xs "results/target_s";
  (* initial guess of inital condition *)
  let x0 =
    Algodiff.D.Maths.concatenate
      ~axis:0
      [| target 0.; Algodiff.D.Mat.gaussian ~sigma:1E-3 (n - n_obs) 1 |]
  in
  (* initial guess of paramter w *)
  let w1 = Algodiff.D.Mat.gaussian ~sigma:1E-3 p n in
  let b1 = Algodiff.D.Mat.gaussian ~sigma:1E-3 p 1 in
  let w2 = Algodiff.D.Mat.gaussian ~sigma:1E-3 n p in
  let b2 = Algodiff.D.Mat.gaussian ~sigma:1E-3 n 1 in
  (* learning x0 and w *)
  let x0, w1, b1, w2, b2 =
    learn
      0
      x0
      w1
      b1
      w2
      b2
      Algodiff.D.Mat.(zeros p n)
      Algodiff.D.Mat.(zeros p 1)
      Algodiff.D.Mat.(zeros n p)
      Algodiff.D.Mat.(zeros n 1)
      1E9
      Algodiff.D.(F 1E-2)
  in
  (* save learnt dynamics *)
  save x0 w1 b1 w2 b2
