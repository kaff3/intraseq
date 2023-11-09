-- ==
-- compiled random input {[100000][1024]i64} auto output
-- compiled random input {[100000][1023]i64} auto output
let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
     map (\a -> a + 2) a_row
  ) a

