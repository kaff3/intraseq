
-- ==
-- entry: fun1
-- compiled random input {[1000][1024]i64} auto output
-- compiled random input {[1000][1023]i64} auto output
entry fun1 [n] [m] (a : [n][m]i64) =
	#[incremental_flattening(only_intra)]
	let res = map (\ a_row -> 
		let tmp = map (\x -> x+2) a_row
		let r2 = scan (*) 1 tmp
		let r1 = reduce (+) 0 r2
		in ((r1, r2), tmp)
	) a
	let (x,y) = unzip res
	let (q,p) = unzip x
	in (q,p,y)

