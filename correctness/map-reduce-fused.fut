-- ==
-- entry: main
-- compiled random input {[1000][1024]i64 [1024]i64} auto output
-- compiled random input {[1000][1023]i64 [1023]i64} auto output
-- compiled random input {[1000][1022]i64 [1022]i64} auto output
-- compiled random input {[1000][1021]i64 [1021]i64} auto output
-- compiled random input {[1000][1]i64 [1]i64} auto output
-- compiled random input {[1000][2]i64 [2]i64} auto output
-- compiled random input {[1000][3]i64 [3]i64} auto output
-- compiled random input {[1000][4]i64 [4]i64} auto output
-- compiled random input {[1000][5]i64 [5]i64} auto output
-- compiled random input {[1000][6]i64 [6]i64} auto output
-- compiled random input {[1000][7]i64 [7]i64} auto output
-- compiled random input {[1000][8]i64 [8]i64} auto output

let main [n] [m] (a : [n][m]i64) (b : [m]i64)  =
  #[incremental_flattening(only_intra)]
  let res = map (\ a_row ->
    let tmp = map (\x -> x / 11) a_row
    let tmp1 = map2 (\x y -> x * y) tmp b
    let r1 = reduce (+) 0 tmp1
    let r2 = reduce (i64.min) i64.highest tmp1
    let tmp2 = map2 (\x y -> x + y) b tmp1
    let tmp3 = map (\x -> x - 99) tmp2
    in (tmp3, (r1, r2))
  ) a
  let (x, y) = unzip res
  let (q, p) = unzip y
  in (x, q, p)
