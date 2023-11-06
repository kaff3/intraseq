
-- ==
-- compiled random input {[100000][1024]i64} auto output
-- compiled random input {[100000][1023]i64} auto output
-- compiled random input {[100000][999]i64} auto output
let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
     reduce (+) 0 a_row
  ) a

