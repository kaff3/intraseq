let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
     scan (*) 1 a_row |> reduce (+) 0
  ) a