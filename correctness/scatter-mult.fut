-- ==
-- entry: main
-- compiled random input {[1000][1024]i16 [1000][1024]i16 } auto output
-- compiled random input {[1000][1023]i16 [1000][1023]i16 } auto output
-- compiled random input {[1000][1022]i16 [1000][1022]i16 } auto output
-- compiled random input {[1000][1021]i16 [1000][1021]i16 } auto output
-- compiled random input {[1000][1]i16 [1000][1]i16 } auto output
-- compiled random input {[1000][2]i16 [1000][2]i16 } auto output
-- compiled random input {[1000][3]i16 [1000][3]i16 } auto output
-- compiled random input {[1000][4]i16 [1000][4]i16 } auto output
-- compiled random input {[1000][5]i16 [1000][5]i16 } auto output
-- compiled random input {[1000][6]i16 [1000][6]i16 } auto output
-- compiled random input {[1000][7]i16 [1000][7]i16 } auto output
-- compiled random input {[1000][8]i16 [1000][8]i16 } auto output

let main [n] [m] (dss: [n][m]i16) (vss: [n][m]i16) = 
	#[incremental_flattening(only_intra)]
	map2 (\ ds vs ->
    let is = iota m 
    let vs' = scan (\x acc -> x + acc) 0 vs
		in scatter (copy ds) is vs'
		) dss vss
	

