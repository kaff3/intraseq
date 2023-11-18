-- ==
-- entry: main
-- compiled random input {[1000][1024]i64} auto output
-- compiled random input {[1000][1023]i64} auto output
-- compiled random input {[1000][1022]i64} auto output
-- compiled random input {[1000][1021]i64} auto output
-- compiled random input {[1000][1]i64} auto output
-- compiled random input {[1000][2]i64 } auto output
-- compiled random input {[1000][3]i64 } auto output
-- compiled random input {[1000][4]i64 } auto output
-- compiled random input {[1000][5]i64 } auto output
-- compiled random input {[1000][6]i64 } auto output
-- compiled random input {[1000][7]i64 } auto output
-- compiled random input {[1000][8]i64 } auto output

let main [n] [m] (a: [m][n]f32) = 
	#[incremental_flattening(only_intra)]
	let res = map (\ row ->
		let row_scanned = scan (+) 0 row
    
		in (reduce (+) 0 row_scanned,
		reduce (f32.min) f32.highest row_scanned)
	) a
	in unzip res
