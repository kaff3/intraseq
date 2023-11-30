let imap as f = 
    #[incremental_flattening(only_intra)]
    -- #[seq_factor(4)]
    map f as


let partition2 [n] (p: u32 -> bool) (arr: [n]u32) 
                : (i64, *[n]u32) =
   let cs = map p arr
   let tfs = map (\ f->if f then 1i64
                           else 0i64) cs
   let isT = scan (+) 0 tfs
   let i = isT[n-1]

   let ffs = map (\f->if f then 0 
                          else 1) cs
   let isF = map (+i) <| scan (+) 0 ffs
   let inds = map (\(c,iT,iF) -> 
                    if c then iT-1 
                         else iF-1
                ) (zip3 cs isT isF)
   let res = replicate n 0
   in (i, scatter res inds arr)

let excl_scan [n] 't (op: t -> t -> t) (ne: t) (arr: [n]t) : [n]t = 
  [ne] ++ (init (scan op ne arr)) :> [n]t


--let hist_op_map (elm: u32) : [16]u32 = 
--    let idx = (elm >> (digit*4)) & 0xF
--    in (replicate 16 0) with [idx] = 1    

let hist_op_red (accum: [16]u32) (elms: [16]u32) : [16]u32 = 
    map2 (\ x y -> x + y) accum elms
    


let step [n] [m] (digit : u32) (arr : *[n][m]u32) : *[n][m]u32 =

    let b : u32 = 4

    -- Rank Kernel
    let (arr', g_hist) = 
        #[incremental_flattening(only_intra)]
        #[seq_factor(4)]
        map ( \ row -> 
            let row' = loop row for j < (i64.u32 b) do
                let (_, row') = partition2 (\ elm ->
                    if (elm >> (digit*b + (u32.i64 j))) & 1 == 0 then true else false
                ) row
                in row'

            let hist_tmp = map ( \ elm ->
                let idx = (elm >> (digit*4)) & 0xF |> i64.u32
                in (replicate 16 0) with [idx] = 1    
            ) row'
            let hist = reduce hist_op_red (replicate 16 0) hist_tmp
            in (row', hist)
    ) arr
    |> unzip

    -- Hist Kernel
    let g_hist' = transpose g_hist
                    |> flatten
                    |> excl_scan (+) 0
                    |> unflatten
                    |> transpose
    let l_hist = 
        #[incremental_flattening(only_intra)]
        #[seq_factor(4)]
        map (\ hist -> excl_scan (+) 0 hist) g_hist

    -- Scatter Kernel
    let idxs =
        #[incremental_flattening(only_intra)]
        #[seq_factor(4)]
         map3 (\ g_hist l_hist row ->
            map2 ( \ elm i ->
                let d = (elm >> (digit*4)) & 0xF
                let g_pos = i64.u32 g_hist[i32.u32 d]
                let l_pos = i - (i64.u32 l_hist[i32.u32 d])
                in g_pos + l_pos
            ) row (iota m)
        ) g_hist' l_hist arr'

    in 
      scatter (flatten arr :> [n*m]u32)
               (flatten idxs :> [n*m]i64)
               (flatten arr' :> [n*m]u32)
        |> unflatten


-- ==
-- entry: main
-- compiled random input {[1000][1024]u32} auto output
let main [n] [m] (arr : *[n][m]u32) : *[n][m]u32 =
    let num_digits = 8
    in loop arr for i < num_digits do
      step (u32.i32 i) arr  
