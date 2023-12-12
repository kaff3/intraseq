let main [n] [m] (a : [n][m]u32) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
     map (\a -> a + 2) a_row
  ) a

-- Map-map-simple performance
-- ==
-- entry: testBlocks
-- compiled random input {[100000][1024]u32} auto output
-- compiled random input {[200000][1024]u32} auto output
-- compiled random input {[300000][1024]u32} auto output
-- compiled random input {[400000][1024]u32} auto output
-- compiled random input {[500000][1024]u32} auto output
-- compiled random input {[600000][1024]u32} auto output
-- compiled random input {[700000][1024]u32} auto output
-- compiled random input {[800000][1024]u32} auto output
-- compiled random input {[900000][1024]u32} auto output
-- compiled random input {[1000000][1024]u32} auto output
let testBlocks [n] [m] (a : [n][m]u32) = main a

-- ==
-- entry: testThreads
-- compiled random input { [131072][1024]u32 }
-- compiled random input { [262144][512]u32 }
-- compiled random input { [524288][256]u32 }
-- compiled random input { [1048576][128]u32 }
let testThreads [n] [m] (a : [n][m]u32) = main a

