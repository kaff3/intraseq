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

let main [n] [m] (a : [n][m]i64) (b:[m]i64) =
  #[incremental_flattening(only_intra)]
  #[seq_factor(4)]
  let res = map (\ a_row ->
   let r1 = scan (+) 0 a_row
	 let tmp1 = map (\x -> x + 3) a_row
	 let tmp2 = map3 (\x y z -> x + y + z) r1 tmp1 b
	 let r2 = reduce (+) 0 tmp2
	 in ((r1,r2), (tmp1, tmp2))
  ) a
  let (x,y) = unzip res
  let (r1,r2) = unzip x
  let (t1,t2) = unzip y
  in (r1,r2,t1,t2)

