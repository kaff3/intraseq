let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map (\ a_row ->
    let tmp = map (\x -> x * 2) a_row
    let tmp1 = map (\x -> x + 2) tmp
    let r1 = reduce (+) 0 tmp
    let r2 = reduce (i64.min) i64.highest tmp
    in (tmp1, r1, r2)
  ) a
