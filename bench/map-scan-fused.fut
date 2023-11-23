-- ==
-- entry: main
-- compiled random input {[100000][1024]i64 [1024]i64} auto output
-- compiled random input {[100000][1023]i64 [1023]i64} auto output
let main [n] [m] (a : [n][m]i64) (b : [m]i64)  =
  #[incremental_flattening(only_intra)]
  let res = map (\ a_row ->
    let tmp = map (\x -> x / 11) a_row
    let tmp1 = map2 (\x y -> x * y) tmp b
    let r1 = scan  (+) 0 tmp1
    let r2 = scan  (i64.min) i64.highest tmp1
    let tmp2 = map2 (\x y -> x + y) b tmp1
    let tmp3 = map (\x -> x - 99) tmp2
    in (tmp3, (r1, r2))
  ) a
  let (x, y) = unzip res
  let (q, p) = unzip y
  in (x, q, p)
