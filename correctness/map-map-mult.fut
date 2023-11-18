-- ==
-- entry: main
-- compiled random input {[1000][1024]i64 [1024]i64} auto output
-- compiled random input {[1000][1023]i64 [1023]i64} auto output
-- compiled random input {[1000][1022]i64 [999]i64} auto output
-- compiled random input {[1000][1021]i64 [999]i64} auto output
-- compiled random input {[1000][1]i64 [1024]i64} auto output
-- compiled random input {[1000][2]i64 [1023]i64} auto output
-- compiled random input {[1000][3]i64 [999]i64} auto output
-- compiled random input {[1000][4]i64 [999]i64} auto output
-- compiled random input {[1000][5]i64 [1024]i64} auto output
-- compiled random input {[1000][6]i64 [1023]i64} auto output
-- compiled random input {[1000][7]i64 [999]i64} auto output
-- compiled random input {[1000][8]i64 [999]i64} auto output
let main [n] [m] (a : [n][m]i64) (b : [m]i64)  =
  #[incremental_flattening(only_intra)]
  let res = map (\ a_row ->
    let tmp = map (\x -> x / 11) a_row
    let tmp1 = map (\x -> x + 32) tmp
    let tmp2 = map (\x -> x * 12) a_row
    in (tmp, (tmp1, tmp2))
  ) a
  let (x, y) = unzip res
  let (q, p) = unzip y
  in (x, q, p)

