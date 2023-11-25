-- ==
-- entry: main
-- compiled random input {[1000][1024]i16 [1000][1024]i64 [1000][1024]i16} auto output
-- compiled random input {[1000][1023]i16 [1000][1023]i64 [1000][1023]i16} auto output
-- compiled random input {[1000][1022]i16 [1000][1022]i64 [1000][1022]i16} auto output
-- compiled random input {[1000][1021]i16 [1000][1021]i64 [1000][1021]i16} auto output
-- compiled random input {[1000][1]i16 [1000][1]i64 [1000][1]i16} auto output
-- compiled random input {[1000][2]i16 [1000][2]i64 [1000][2]i16} auto output
-- compiled random input {[1000][3]i16 [1000][3]i64 [1000][3]i16} auto output
-- compiled random input {[1000][4]i16 [1000][4]i64 [1000][4]i16} auto output
-- compiled random input {[1000][5]i16 [1000][5]i64 [1000][5]i16} auto output
-- compiled random input {[1000][6]i16 [1000][6]i64 [1000][6]i16} auto output
-- compiled random input {[1000][7]i16 [1000][7]i64 [1000][7]i16} auto output
-- compiled random input {[1000][8]i16 [1000][8]i64 [1000][8]i16} auto output

let main [n] [m] (dss: [n][m]i16) (iss: [n][m]i64) (vss: [n][m]i16) = 
	#[incremental_flattening(only_intra)]
  #[seq_factor(4)]
	map3 (\ ds is vs ->
		scatter (copy ds) is vs
		) dss iss vss
