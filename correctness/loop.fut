-- ==
-- entry: main
-- compiled random input {[1000][1024]i32 1i64 } auto output
-- compiled random input {[1000][1023]i32 1i64 } auto output
-- compiled random input {[1000][1022]i32 1i64 } auto output
-- compiled random input {[1000][1021]i32 1i64 } auto output
-- compiled random input {[1000][1]i32 1i64 } auto output
-- compiled random input {[1000][2]i32 1i64 } auto output
-- compiled random input {[1000][3]i32 1i64 } auto output
-- compiled random input {[1000][4]i32 1i64 } auto output
-- compiled random input {[1000][5]i32 1i64 } auto output
-- compiled random input {[1000][6]i32 1i64 } auto output
-- compiled random input {[1000][7]i32 1i64 } auto output
-- compiled random input {[1000][8]i32 1i64 } auto output
let main [n] [m] (ass: *[n][m]i32) (k:i64) =
	#[incremental_flattening(only_intra)]
	#[seq_factor(4)]
	let result = map (\ as ->
		let res = loop as for i < k do
						let tmp = map (\ a -> a+2) as
						in tmp
		in res
	) ass
	in result
