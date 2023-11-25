-- ==
-- entry: main
-- compiled random input {[1000][1024]f32} auto output
-- compiled random input {[1000][1023]f32} auto output
-- compiled random input {[1000][1022]f32} auto output
-- compiled random input {[1000][1021]f32} auto output
-- compiled random input {[1000][1]f32} auto output
-- compiled random input {[1000][2]f32 } auto output
-- compiled random input {[1000][3]f32 } auto output
-- compiled random input {[1000][4]f32 } auto output
-- compiled random input {[1000][5]f32 } auto output
-- compiled random input {[1000][6]f32 } auto output
-- compiled random input {[1000][7]f32 } auto output
-- compiled random input {[1000][8]f32 } auto output

let main [n] [m] (a: [m][n]f32) = 
	#[incremental_flattening(only_intra)]
  #[seq_factor(4)]
	let res = map (\ row ->
		let row_scanned = scan (+) 0 row
    
		in (reduce (+) 0 row_scanned,
		reduce (f32.min) f32.highest row_scanned)
	) a
	in unzip res
