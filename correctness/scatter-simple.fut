

-- ==
-- compiled random input {[100][1024]i16 [100][1024]i64 [100][1024]i16} auto output
-- compiled random input {[100][1023]i16 [100][1023]i64 [100][1023]i16} auto output
-- compiled random input {[100][999]i16  [100][999]i64  [100][999]i16} auto output
let main [n] [m] (dss: [n][m]i16) (iss: [n][m]i64) (vss: [n][m]i16) = 
	#[incremental_flattening(only_intra)]
	map3 (\ ds is vs ->
		scatter (copy ds) is vs
		) dss iss vss
	
