let main [n] [m] (a : [n][m]i64) =
  map (\ a' ->
     map (\ a'' -> 
      let arr = iota a'' in
      scan (\ x y -> x + y) 0 arr
      |> reduce (+) 0     
    ) a'
  ) a