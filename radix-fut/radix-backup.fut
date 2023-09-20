let cmap as f = map f as

-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}
let step [num_blocks] [num_threads] [e] (digit : u32) (arr : *[num_blocks][num_threads*e]u32)
          : *[num_blocks][2**4]u32 =
    -- let b_splits = 4
    let b = 4
    -- let arr_flat = flatten arr

    -- let g_hist = replicate (2**b_splits * num_blocks) 0u32
    --           |> unflatten num_blocks (2**b_splits)
    
    -- rankKernel
    let (g_hist, arr) = cmap (iota num_blocks)
        ( \ blkid ->
            let sh_hist = replicate (2**b) 0u32
            -- let sh_tile = replicate num_threads (replicate e 0u32)
            let num_threads_iota = iota num_threads

            -- Load the tile 
            -- let tmp = map (\ x -> sh_tile with [0 : x*22] = ...) (iota ?)
            -- map (\ local_tile -> local_tile with )

            -- Load the tile. Each thread gets one chunk of size e
            -- let sh_tile = map (\_ ->
            --     -- loop chunk for i < e do
            --     --     chunk with [i] = arr[blkid][i] 
            --     arr[blkid]
            -- ) sh_tile

            -- Load tile
            let sh_tile = unflatten arr[blkid] -- what

            -- let _ =
            --     (map (\ tid -> 
            --         let sh_tile = loop sh_tile for k < e do
            --            sh_tile with [tid*e + k] = arr[blkid][tid*e + k]
            --         in 0
            --     ) num_threads_iota)
            
            -- let sh_tile =
            --     map (\ tid ->
            --         let tmp = arr[blkid]
            --         in loop sh_tile for k < e do
            --             sh_tile with [tid*e + k] = tmp[tid*e + k]                
            --     ) num_threads_iota


            -- Local sort, bit iterations of 1-bit split
            -- let psss = replicate num_threads (0,0)
            
            let psss = map (\ chunk -> 
                map (\i ->
                    let ps0 = reduce (\ps0 x -> 
                        let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                        in ps0 + bit^1
                        ) 0 chunk
                    let ps1 = reduce (\ps1 x -> 
                        let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                        in ps1 + bit
                        ) 0 chunk 
                    in [ps0, ps1]
                ) (iota 4)
            ) sh_tile

            
            let sh_tile = map (\ tid ->
                let sh_tile = loop sh_tile for i < b do
                    let chunk = sh_tile[tid]
                    -- thread local histogram
                    --let ps0 = reduce (\ps0 x -> 
                    --    let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                    --    in ps0 + bit^1
                    --    ) 0 chunk
                    --let ps1 = reduce (\ps1 x -> 
                    --    let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                    --    in ps1 + bit
                    --    ) 0 chunk 

                    -- let psss = psss with [tid] = (ps0, ps1)
                    
                    --

                    -- scan 
                    let agg = reduce (\ (a0, _) (x, _) -> (0, a0+x)) (0,0) psss[i]
                    let sc = scan (\(e1,e2) (acc1, acc2) -> (e1 + acc1, e2 + acc2)) agg psss[i]
                    let sc_excl = [(0,0)] ++ init sc 
                    
                    -- scatter
                    let idxs = replicate e (0i64, 0i64)             
                    let (_, idxs) = loop ((ps0, ps1), idxs) = (sc_excl[tid], idxs) for x < e do
                              let ele = sh_tile[tid][x]
                              let bit = ((ele >> (digit*4 + (u32.i64 i))) & 1)
                              let idxs = idxs with [tid*x] = if bit == 0 then (tid, i64.u32 ps0) else (tid, i64.u32 ps1)
                              in ((ps0 + bit^1, ps1 + bit), idxs)

                    in scatter_2d sh_tile idxs chunk

                in sh_tile[tid]
            ) num_threads_iota


            -- blockwide hist
            -- !!!!!SEQ!!!!! fix pls
            let tmp = flatten sh_tile
            let sh_hist = loop sh_hist for i < e*num_threads do
                let ele = tmp[i]
                let dig = i64.u32 ((ele >> (digit * 4)) & 0xFF)
                in sh_hist with [dig] = sh_hist[dig] + 1

            -- write back sorted tile
            -- let _ = cmap (num_threads_iota) (\ tid ->
            --     let arr_flat = loop arr_flat for k < e do
            --         arr_flat with [blkid * tid*e + k] = sh_tile[tid*e + k]
            --     in tid
            -- )

            -- write back sorted tile
            -- let arr = map (\chunk -> 
            --    arr with [blkid] = chunk
            -- ) sh_tile

            -- let arr[blkid] = sh_tile

            -- let arr = scatter arr idxs ??

            in (sh_hist, sh_tile)
        ) |> unzip

        in g_hist


