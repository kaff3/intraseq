
-- ==
-- entry: fun1
-- compiled random input {[1000][1024]i64} auto output
-- compiled random input {[1000][1023]i64} auto output
entry fun1 [n] [m] (a : [n][m]i64) =
	#[incremental_flattening(only_intra)]
	let res = map (\ a_row -> 
		let r2 = scan (*) 1 a_row
		let r1 = reduce (+) 0 r2
		in (r1, r2)
	) a
	in unzip res


