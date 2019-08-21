open Owl

let print_dim x =
  let d1, d2 = Mat.shape x in
  Printf.printf "%i, %i\n%!" d1 d2


let print_algodiff_dim x =
  let dims = Algodiff.D.shape x in
  let d1 = dims.(0)
  and d2 = dims.(1) in
  Printf.printf "%i, %i\n%!" d1 d2


let save_bin filename m =
  let output = open_out filename in
  Marshal.to_channel output m [ Marshal.No_sharing ];
  close_out output


let read_bin filename =
  let input = open_in filename in
  let m = Marshal.from_channel input in
  close_in input;
  m
