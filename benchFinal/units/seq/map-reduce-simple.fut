let main [n] [m] (a : [n][m]u32) =
  #[incremental_flattening(only_intra)]
  #[seq_factor(4)]
  map (\ a_row ->
     reduce (+) 0 a_row
  ) a

-- Map-reduce-simple performance
-- ==
-- entry: testBlocks
-- compiled random input {[100000][1024]u32} auto output
-- compiled random input {[200000][1024]u32} auto output
-- compiled random input {[300000][1024]u32} auto output
-- compiled random input {[400000][1024]u32} auto output
-- compiled random input {[500000][1024]u32} auto output
entry testBlocks [n] [m] (a : [n][m]u32) = main a

-- ==
-- entry: testThreads
-- compiled random input { [32768][4096]u32 }
-- compiled random input { [65536][2048]u32 }
-- compiled random input { [131072][1024]u32 }
-- compiled random input { [262144][512]u32 }
-- compiled random input { [524288][256]u32 }
-- compiled random input { [1048576][128]u32 }
entry testThreads [n] [m] (a : [n][m]u32) = main a

