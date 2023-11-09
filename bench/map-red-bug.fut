-- ==
-- compiled random input {[100000][1024]i64 [1024]i64} auto output
-- compiled random input {[100000][1023]i64 [1023]i64} auto output
let main [n] [m] (a : [n][m]i64) (b:[m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
	 let tmp1 = map (\x -> x + 3) a_row
	 let r2 = reduce (+) 0 tmp1
	 in (tmp1, r2)
  ) a


