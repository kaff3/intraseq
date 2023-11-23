let main [n] [m] (a : [n][m]i64) =
  #[incremental_flattening(only_intra)]
  map(\a_row -> 
    let local_reds = 
      map(\tid ->
        let chunk = a_row[tid * 4 : tid * 4 + 4] 
        in loop v = 0 for i < 4 do
          v + chunk[i]
      ) (iota (m / 4))
    in reduce (+) 0 local_reds
  ) a 
