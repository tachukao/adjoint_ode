module type PT = sig
  val dim : int
  val fd : Owl.Algodiff.D.t -> float -> Owl.Algodiff.D.t
  val t0 : float
  val t1 : float
  val dt : float
end

module Make (P : PT) = struct
  include P
  open Owl
  open Owl_ode
  open Owl_ode_odepack

  (** define the dynamical system equation dx/dt = f(x,t) using
 * the differentiable version fd(x,t). Only difference between
 * f(x,t) and fd(x,t) here is the x's type *)
  let f x t =
    let x = Algodiff.D.pack_arr x in
    let dxdt = fd x t in
    Algodiff.D.unpack_arr dxdt


  (** define the corresponding adjoint dynamical system 
 * ds/dt = adj_f(s,t). The adjoint state s = [ x @= a ]
 * is a vertical concatenation of x and a, where a = dl/dx 
 * is the derivative of the loss l with respect to the state
 * x *)
  let adj_f =
    let extract s =
      let x = Mat.(s.${[ [ 0; pred dim ] ]}) in
      let a = Mat.(s.${[ [ dim; -1 ] ]}) in
      x, a
    in
    fun s t ->
      let x, a = extract s in
      let x = Algodiff.D.pack_arr x in
      let a = Algodiff.D.pack_arr a in
      let dx = Algodiff.D.Maths.neg (fd x t) in
      let da = Algodiff.D.(Maths.neg (jacobianTv (fun x -> fd x t) x a)) in
      let dx = Algodiff.D.unpack_arr dx in
      let da = Algodiff.D.unpack_arr da in
      Owl.Mat.(dx @= da)


  (* time specification *)
  let duration = t1 -. t0
  let dt = dt (*duration*)
  let tspec = Types.(T1 { t0; dt; duration })

  (* forward pass through time *)
  let forward x0 =
    let _, xs = Ode.odeint (module Lsoda) f x0 tspec () in
    Mat.col xs (-1)


  (* backward pass through time *)
  let backward ~loss =
    let dloss = Algodiff.D.grad loss in
    fun x1 ->
      let l = loss (Algodiff.D.pack_arr x1) |> Algodiff.D.unpack_flt in
      let s1 =
        let a1 = dloss Algodiff.D.(pack_arr x1) |> Algodiff.D.unpack_arr in
        Mat.(x1 @= a1)
      in
      let tspec = Types.(T1 { t0 = t1; dt = -.dt; duration = -.duration }) in
      let _, adj_ss = Ode.odeint (module Lsoda) adj_f s1 tspec () in
      Mat.col adj_ss (-1), l


  (* gradient at x0 : returns dx0 and loss *)
  let grad ~loss =
    let backward = backward ~loss in
    fun x0 ->
      (* run the dynamics forward from current guess of x0 *)
      let x1 = forward x0 in
      (* calculate the loss l and its gradient with respect to x0 *)
      let s0, l = backward x1 in
      Mat.(s0.${[ [ dim; -1 ] ]}), l
end
