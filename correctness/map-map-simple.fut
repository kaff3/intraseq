-- ==
-- entry: main
-- compiled random input {[1000][1024]i64} auto output
-- compiled random input {[1000][1023]i64} auto output
-- compiled random input {[1000][1022]i64} auto output
-- compiled random input {[1000][1021]i64} auto output
-- compiled random input {[1000][1]i64} auto output
-- compiled random input {[1000][2]i64 } auto output
-- compiled random input {[1000][3]i64 } auto output
-- compiled random input {[1000][4]i64 } auto output
-- compiled random input {[1000][5]i64 } auto output
-- compiled random input {[1000][6]i64 } auto output
-- compiled random input {[1000][7]i64 } auto output
-- compiled random input {[1000][8]i64 } auto output

let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
     map (\a -> a + 2) a_row
  ) a

