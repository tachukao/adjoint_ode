open Owl
open Owl_ode

(* dimension of state-space *)
let n = 4

(* helper function *)
let print_dim x =
  let dims = Algodiff.D.shape x in
  let d1 = dims.(0)
  and d2 = dims.(1) in
  Printf.printf "%i, %i\n%!" d1 d2


(* herlper module: custom solver that does the 
 * unpacking and packing of Algodiff arrays 
 * automatically: the states and functions have 
 * type Algodiff.D.t but the outputs are Mat.mat *)
module CSolver = struct
  type state = Algodiff.D.t
  type f = Algodiff.D.t -> float -> Algodiff.D.t
  type step_output = Mat.mat * float
  type solve_output = Mat.mat * Mat.mat

  let step f ~dt y0 t0 =
    let y0 = Algodiff.D.unpack_arr y0 in
    let f y (t : float) =
      let y = Algodiff.D.pack_arr y in
      f y t |> Algodiff.D.unpack_arr
    in
    Owl_ode_odepack.lsoda_s ~relative_tol:1E-3 ~abs_tol:1E-3 f ~dt y0 t0


  let solve f y0 tspec () =
    let y0 = Algodiff.D.unpack_arr y0 in
    let f y (t : float) =
      let y = Algodiff.D.pack_arr y in
      f y t |> Algodiff.D.unpack_arr
    in
    Owl_ode_odepack.lsoda_i ~relative_tol:1E-3 ~abs_tol:1E-3 f y0 tspec ()
end

(* time specification *)
let t0 = 0.
let t1 = 2.
let duration = t1 -. t0
let dt = t1
let tspec = Types.(T1 { t0; dt; duration })

(* define the dynamical system model: 
 * dx/dt = f(x,t) = w *@ x, where w is 
 * a matrix paramter that we will also 
 * learn *)
let g w x = Algodiff.D.Maths.(w *@ x)
let f w x _t = g w x

(* the adjoint state space is a concatenation of the current state x
 * derivative of the loss with respective to x and the derivation
 * of the loss with respect to parameters w *)
let extract s =
  let open Algodiff.D in
  let x = Maths.get_slice [ [ 0; pred n ] ] s in
  let dldx = Maths.get_slice [ [ n; pred (2 * n) ] ] s in
  let dldw = Maths.get_slice [ [ 2 * n; -1 ] ] s in
  x, dldx, dldw


(* define the adjoint dynamical system: 
 * ds/dt = adjf(s,w,t) = 
 * [-dx/dt; d(dldx)/dt; d(dl/dw)/dt] *)
let adjf w s _t =
  let open Algodiff.D in
  let x, dldx, _ = extract s in
  let w = primal w in
  let dx = Maths.neg (g w x) |> primal in
  let g1 x = g w x in
  let g2 w = g w x in
  let ddldx = Maths.neg (jacobianTv g1 x dldx) |> primal in
  let ddldw = jacobianTv g2 w dldx |> fun x -> Maths.reshape x [| -1; 1 |] |> primal in
  Maths.concatenate ~axis:0 [| dx; ddldx; ddldw |]


(* learnig rate and maximum gradient iterations *)
let alpha = 1E-2
let max_iter = 10000

(* target at time t *)
let target =
  let omega = 1.5 in
  fun t ->
    [| Maths.sin (omega *. t); Maths.cos (omega *. t); t; -.t -. 1. |]
    |> fun x -> Owl.Mat.of_array x n 1 |> Algodiff.D.pack_arr


(* loss with respect to state x at time t *)
let loss t =
  let target = target t in
  fun x -> Algodiff.D.Maths.(l2norm_sqr' (x - target))


(* derivative of loss with repsect to x at time t *)
let dloss t = Algodiff.D.grad (loss t)

(* running the dynamics forward in time *)
let forward w x0 =
  let tspec = Types.(T1 { t0; dt; duration }) in
  let _, xs = Ode.odeint (module CSolver) (f w) x0 tspec () in
  Mat.col xs (-1) |> Algodiff.D.pack_arr


(* running the dynamics backward in time and calculating gradients *)
let backward w x1 =
  (* times at which to evaluate loss and gradients *)
  let dt = 1E-1 in
  let adjf = adjf w in
  let open Algodiff.D in
  let s1 =
    let dldx = dloss t1 x1 |> primal in
    Maths.concatenate ~axis:0 [| x1; Maths.(F alpha * dldx); Mat.(zeros (n * n) 1) |]
  in
  (* evaluate and accumulate gradients at intermediate time t *)
  let rec accu s t l =
    if t < t0
    then (
      let x0, dldx, dldw = extract s in
      let dldw = Maths.reshape dldw [| n; n |] in
      let dldw = dldw in
      l, x0, dldx, dldw)
    else (
      let tspec = Types.(T1 { t0; dt; duration = dt }) in
      let _, adjs = Ode.odeint (module CSolver) adjf s tspec () in
      let s0 = Owl.Mat.col adjs (-1) |> pack_arr in
      let t = t -. dt in
      let x, dldx, dldw = extract s0 in
      let dldx = Maths.(dldx + (F alpha * dloss t x)) |> primal in
      let s0 = Maths.concatenate ~axis:0 [| x; dldx; dldw |] in
      let l = l +. (loss t x |> unpack_flt) in
      accu s0 t l)
  in
  accu s1 t1 0.


(* helper function to save xs *)
let save_xs x0 w =
  let tspec = Types.(T1 { t0; dt = 1E-2; duration }) in
  let ts, xs = Ode.odeint (module CSolver) (f w) x0 tspec () in
  Owl.Mat.save_txt Owl.Mat.(transpose (ts @= xs)) "actual_xs"


(* gradient + RMSprop to speed up learning *)
let rec learn step x0 w =
  if step < max_iter
  then (
    let open Algodiff.D in
    let x1 = forward w x0 in
    let l, _, dldx0, dldw = backward w x1 in
    let w = Maths.(w - (F alpha * dldw)) in
    let x0 = Maths.(x0 - (F alpha * dldx0)) in
    if step mod 10 = 0 then Printf.printf "\rstep %i | loss %4.5f%!" step l;
    (* save the dynamics every 100 gradient steps *)
    if step mod 100 = 0 then save_xs x0 w;
    learn (succ step) x0 w)
  else x0, w


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
  let x0 = Mat.of_array [| 0.1; -0.1; -0.2; -0.5 |] n 1 |> Algodiff.D.pack_arr in
  (* initial guess of paramter w *)
  let w = Algodiff.D.Mat.uniform ~a:(-0.01) ~b:0.01 n n in
  (* learning x0 and w *)
  let x0, w = learn 0 x0 w in
  (* save learnt dynamics *)
  save_xs x0 w
