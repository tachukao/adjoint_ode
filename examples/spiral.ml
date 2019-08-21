(* code ported from https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py *)

open Owl

let a = 0.
let b = 0.3
let t0 = 0.
let t1 = 6. *. Const.pi
let n_total = 500
let sample_every = 5
let orig_ts = Mat.linspace t0 t1 n_total |> Mat.transpose
let samp_ts = Mat.get_slice [ [ 0; -1; sample_every ] ] orig_ts
let n_samples = Mat.numel samp_ts
let noise_std = 0.3
let n_spirals = 100

(* clock-wise *)
let cw_traj =
  let z = Mat.(t1 +. 1. $- orig_ts) in
  let r = Mat.((b *. 50. $/ z) +$ a) in
  let xs = Mat.((r * cos z) -$ 5.) in
  let ys = Mat.(r * sin z) in
  Mat.(xs @|| ys)


let cc_traj =
  let z = orig_ts in
  let r = Mat.((b $* z) +$ a) in
  let xs = Mat.((r * cos z) +$ 5.) in
  let ys = Mat.(r * sin z) in
  Mat.(xs @|| ys)


let generate =
  let n = n_total - (2 * n_samples) in
  let p = Array.make n (1. /. float n) in
  fun () ->
    let t0_idx =
      let i = Stats.multinomial_rvs ~p 1 |> Array.map float |> Stats.max_i in
      i + n_samples
    in
    let orig_traj = if Random.bool () then cc_traj else cw_traj in
    let samp_traj =
      let x = Mat.get_slice [ [ t0_idx; pred t0_idx + n_samples ] ] orig_traj in
      Mat.(x + gaussian ~sigma:noise_std n_samples 2)
    in
    orig_traj, samp_traj


let data =
  let data = Array.init n_spirals (fun _ -> generate ()) in
  let orig = Array.map fst data in
  let samp = Array.map snd data in
  orig, samp, orig_ts, samp_ts


let save_bin filename m =
  let output = open_out filename in
  Marshal.to_channel output m [ Marshal.No_sharing ];
  close_out output


(* reads whatever was saved using [save_bin] *)
let read_bin filename =
  let input = open_in filename in
  let m = Marshal.from_channel input in
  close_in input;
  m


let () =
  Mat.save_txt cw_traj "dataset/cw_spiral";
  Mat.save_txt cc_traj "dataset/cc_spiral";
  save_bin "dataset/spiral_dataset" data
