let main [n] [m] (dss: [n][m]u32) (vss: [n][m]u32) = 
	#[incremental_flattening(only_intra)]
	map2 (\ ds vs ->
		scatter (copy ds) (iota m) vs
		) dss vss

-- Map-scatter-simple performance
-- ==
-- entry: testBlocks
-- compiled random input {[50000][1024]u32 [50000][1024]u32} auto output
-- compiled random input {[100000][1024]u32 [100000][1024]u32} auto output
-- compiled random input {[150000][1024]u32 [150000][1024]u32} auto output
-- compiled random input {[200000][1024]u32 [200000][1024]u32} auto output
-- compiled random input {[250000][1024]u32 [250000][1024]u32} auto output
entry testBlocks [n] [m] (a : [n][m]u32) (c : [n][m]u32) = main a c

-- ==
-- entry: testThreads
-- compiled random input { [131072][1024]u32  [131072][1024]u32 } auto output
-- compiled random input { [262144][512]u32  [262144][512]u32 } auto output
-- compiled random input { [524288][256]u32  [524288][256]u32 } auto output
-- compiled random input { [1048576][128]u32  [1048576][128]u32 } auto output
entry testThreads [n] [m] (a : [n][m]u32) (c : [n][m]u32) = main a c



