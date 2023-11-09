

-- ==
-- compiled random input {[1000][1024]i64 [1024]i64} auto output
-- compiled random input {[1000][1023]i64 [1023]i64} auto output
-- compiled random input {[1000][999]i64 [999]i64} auto output
let main [n] [m] (a : [n][m]i64) (b:[m]i64) =
  #[incremental_flattening(only_intra)]
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

