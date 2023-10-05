
let main [n] [m] (a : [n][m]i32) = 
	#[incremental_flattening(only_intra)]
	map ( \ a_row ->
		map (+1) a_row |> map (*2) |> scan (+) 0 |> reduce (+) 0
	) a