let main [n] [m] (dss: [n][m]i16) (iss: [n][m]i64) (vss: [n][m]i16) = 
	#[incremental_flattening(only_intra)]
  #[seq_factor(4)]
	map3 (\ ds is vs ->
		scatter (copy ds) is vs
		) dss iss vss

-- Map-scatter-simple performance
-- ==
-- entry: testBlocks
-- compiled random input {[100000][1024]u32 [100000][1024]i64 [100000][1024]u32}} auto output
-- compiled random input {[200000][1024]u32 [200000][1024]i64 [200000][1024]u32}} auto output
-- compiled random input {[300000][1024]u32 [300000][1024]i64 [300000][1024]u32}} auto output
-- compiled random input {[400000][1024]u32 [400000][1024]i64 [400000][1024]u32}} auto output
-- compiled random input {[500000][1024]u32 [500000][1024]i64 [500000][1024]u32}} auto output
-- compiled random input {[600000][1024]u32 [600000][1024]i64 [600000][1024]u32}} auto output
-- compiled random input {[700000][1024]u32 [700000][1024]i64 [700000][1024]u32}} auto output
-- compiled random input {[800000][1024]u32 [800000][1024]i64 [800000][1024]u32}} auto output
-- compiled random input {[900000][1024]u32 [900000][1024]i64 [900000][1024]u32}} auto output
-- compiled random input {[1000000][1024]u32 [1000000][1024]i64 [1000000][1024]u32}} auto output
let testBlocks [n] [m] (a : [n][m]u32) = main a

-- ==
-- entry: testThreads
-- compiled random input { [65536][2048]u32 [65536][2048]i64 [65536][2048]u32 }
-- compiled random input { [131072][1024]u32  [131072][1024]i64  [131072][1024]u32 }
-- compiled random input { [262144][512]u32  [262144][512]i64  [262144][512]u32 }
-- compiled random input { [524288][256]u32  [524288][256]i64  [524288][256]u32 }
-- compiled random input { [1048576][128]u32  [1048576][128]i64  [1048576][128]u32 }
let testThreads [n] [m] (a : [n][m]u32) = main a


