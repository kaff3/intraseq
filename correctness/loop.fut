

let main [n] [m] (ass: *[n][m]i32) (k:i64) =
	#[incremental_flattening(only_intra)]
	#[seq_factor(4)]
	let result = map (\ as ->
		let res = loop as for i < k do
						let tmp = map (\ a -> a+2) as
						in tmp
		in res
	) ass
	in result
