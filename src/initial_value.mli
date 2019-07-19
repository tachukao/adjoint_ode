open Owl

module type PT = sig
  val dim : int
  val fd : Owl.Algodiff.D.t -> float -> Owl.Algodiff.D.t
  val t0 : float
  val t1 : float
end

module Make (P : PT) : sig
  include PT

  val duration : float

  (** define the dynamical system equation dx/dt = f(x,t) using
 * the differentiable version fd(x,t). Only difference between
 * f(x,t) and fd(x,t) here is the x's type *)
  val f : Mat.mat -> float -> Mat.mat

  (** define the corresponding adjoint dynamical system 
 * ds/dt = adj_f(s,t). The adjoint state s = [ x @= a ]
 * is a vertical concatenation of x and a, where a = dl/dx 
 * is the derivative of the loss l with respect to the state
 * x *)
  val adj_f : Mat.mat -> float -> Mat.mat

  (** tspect *)
  val tspec : Owl_ode.Types.tspec

  (* forward pass through time *)
  val forward : Mat.mat -> Mat.mat

  (* backward pass through time *)
  val backward : loss:(Algodiff.D.t -> Algodiff.D.t) -> Mat.mat -> Mat.mat * float

  (* gradient at x0 : returns dx0 and loss *)
  val grad : loss:(Algodiff.D.t -> Algodiff.D.t) -> Mat.mat -> Mat.mat * float
end
