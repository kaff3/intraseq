entry main [n] [m] (a: [m][n]f32) = 
	#[incremental_flattening(only_intra)]
	let res = map (\ row ->
		let row_scanned = scan (+) 0 row
    
		in (reduce (+) 0 row_scanned,
		reduce (f32.min) f32.highest row_scanned)
	) a
	in unzip res
