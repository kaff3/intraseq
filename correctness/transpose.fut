let seqmap f x = 
	#[incremental_flattening(only_intra)]
	#[seq_factor(4)]
	map f x

-- ==
-- entry: main
-- compiled random input {[2][16]i32} auto output

-- compiled random input {[1000][1024]i32} auto output
-- compiled random input {[1000][1023]i32} auto output
-- compiled random input {[1000][1022]i32} auto output
-- compiled random input {[1000][1021]i32} auto output
-- compiled random input {[1000][1]i32} auto output
-- compiled random input {[1000][2]i32 } auto output
-- compiled random input {[1000][3]i32 } auto output
-- compiled random input {[1000][4]i32 } auto output
-- compiled random input {[1000][5]i32 } auto output
-- compiled random input {[1000][6]i32 } auto output
-- compiled random input {[1000][7]i32 } auto output
-- compiled random input {[1000][8]i32 } auto output
let main [n] [m] (ass: [n][m]i32) = 
	transpose ass
	
	--let ass' = seqmap (\as -> scan (+) 0 as) ass
	--let bss = transpose ass'
	--in seqmap (\bs -> reduce (+) 0 bs) bss
		
