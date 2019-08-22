open Owl

(* herlper module: custom solver that does the 
   unpacking and packing of Algodiff arrays 
   automatically: the states and functions have 
   type Algodiff.D.t but the outputs are Mat.mat *)

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
    Owl_ode_odepack.lsoda_s ~relative_tol:1E-7 ~abs_tol:1E-9 f ~dt y0 t0


  let solve f y0 tspec () =
    let y0 = Algodiff.D.unpack_arr y0 in
    let f y (t : float) =
      let y = Algodiff.D.pack_arr y in
      f y t |> Algodiff.D.unpack_arr
    in
    Owl_ode_odepack.lsoda_i ~relative_tol:1E-7 ~abs_tol:1E-9 f y0 tspec ()
end
