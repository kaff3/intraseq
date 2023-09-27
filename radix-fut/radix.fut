let imap as f = map f as

-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}

-- num_elems = e * num_threads
let step [num_blocks] [num_elems] (num_threads : i64) (e : i64) (digit : u32) 
         (arr : *[num_blocks][num_elems]u32) : *[num_blocks][]u32 =

    -- rankkernel
    let arr = imap (iota num_blocks)
    ( \ blkid ->

        -- then number of splits
        let b = 4

        -- Load the tile
        let sh_tile = arr[blkid]

        in loop sh_tile for k < b do

            -- Compute indices used for scatter
            
            -- sh_hist_bit is a collective histogram over the amount of bits set and unset 
            -- in each chunk
            let sh_hist_bit = imap (iota num_threads)
            ( \ tid ->
                -- the elements the thread should work on
                let chunk = sh_tile[tid*e : (tid+1)*e]

                let hist = 
                    loop (b0, b1) = (0,0) for i < e do
                        let elem = chunk[i]
                        let bit = u32.get_bit k elem
                        in if bit == 0 then
                            (b0+1, b1)
                        else
                            (b0, b1+1)

                in hist
            )
            |> unzip
            |> (\ (b0s, b1s) -> 
                let b0s_red = reduce (+) 0 b0s
                let b0s_scan = scan (+) 0 b0s       |> init |> (\ x -> [0] ++ x)
                let b1s_scan = scan (+) b0s_red b1s |> init |> (\ x -> [b0s_red] ++ x)
                in zip b0s_scan b1s_scan
            )

            -- Now each thread can use the now scanned histogram to compute block local
            -- indices for all of its elements
            let idxs = imap (iota num_threads)
            (\ tid ->
                let chunk = sh_tile[tid*e : (tid+1)*e]
                let hist = sh_hist_bit[tid]

                let (idxs, _) =
                    loop (idxs, (b0,b1)) = (replicate e 0i64, hist) for i < e do
                        let elem = chunk[i]
                        let bit = u32.get_bit k elem
                        in if bit == 0 then 
                            (idxs with [i] = b0, (b0+1, b1))
                        else
                            (idxs with [i] = b1, (b0, b1+1))
                in idxs
            )
            |> flatten :> [num_elems]i64
            in scatter (copy sh_tile) idxs sh_tile

            -- let sh_tile' = 
            --     loop sh_tile' = copy sh_tile for i < e do
            --         scatter sh_tile' idxs[i] sh_tile
            
            -- in sh_tile'
    )


    in arr 


