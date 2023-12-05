let seqmap f x = 
	#[incremental_flattening(only_intra)]
	#[seq_factor(4)]
	map f x

-- ==
-- entry: main
-- compiled random input {[1000][1024]i16} auto output
-- compiled random input {[1000][1023]i16} auto output
-- compiled random input {[1000][1022]i16} auto output
-- compiled random input {[1000][1021]i16} auto output
-- compiled random input {[1000][1]i16} auto output
-- compiled random input {[1000][2]i16 } auto output
-- compiled random input {[1000][3]i16 } auto output
-- compiled random input {[1000][4]i16 } auto output
-- compiled random input {[1000][5]i16 } auto output
-- compiled random input {[1000][6]i16 } auto output
-- compiled random input {[1000][7]i16 } auto output
-- compiled random input {[1000][8]i16 } auto output
let main [n] [m] (ass: [n][m]i32) = 
	let ass' = seqmap (\as -> scan (+) 0 as) ass
	let bss = transpose ass'
	in seqmap (\bs -> reduce (+) 0 bs) bss
		
