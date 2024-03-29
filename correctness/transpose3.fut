

let seqmap2 f x y = 
	#[incremental_flattening(only_intra)]
	#[seq_factor(4)]
	map2 f x y

-- ==
-- entry: main
-- compiled random input {[1000][1024]i32 [1024][1000]i32} auto output
-- compiled random input {[1000][1023]i32 [1023][1000]i32} auto output
-- compiled random input {[1000][1022]i32 [1022][1000]i32} auto output
-- compiled random input {[1000][1021]i32 [1021][1000]i32} auto output
-- compiled random input {[1000][1]i32 [1][1000]i32} auto output
-- compiled random input {[1000][2]i32 [2][1000]i32 } auto output
-- compiled random input {[1000][3]i32 [3][1000]i32 } auto output
-- compiled random input {[1000][4]i32 [4][1000]i32 } auto output
-- compiled random input {[1000][5]i32 [5][1000]i32 } auto output
-- compiled random input {[1000][6]i32 [6][1000]i32 } auto output
-- compiled random input {[1000][7]i32 [7][1000]i32 } auto output
-- compiled random input {[1000][8]i32 [8][1000]i32 } auto output
let main [n] [m] (ass: [n][m]i32) (bss: [m][n]i32) = 
	let ass' = transpose ass
	let bss' = seqmap2 (\as' bs -> 
		let x = reduce (+) 0 as'
		let y = reduce (+) 0 bs
		in (x,y)
		) ass' bss
	in unzip bss'		
