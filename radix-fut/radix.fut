let cmap as f = map f as

-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}
let step [num_blocks] (e : i64) (digit : u32) (num_threads : i64) (arr : [num_blocks][]u32)
          : [2**4]u32 =
    let b_splits = 4
    let b = 4
    
    let g_hist = replicate (2**b_splits * num_blocks) 0u32
              |> unflatten num_blocks (2**b_splits)
    
    -- rankKernel
    cmap (iota num_blocks)
        ( \ blkid ->
            let sh_hist = replicate (2**b) 0u32
            let sh_tile = replicate (e*num_threads) 0u32
            let num_threads_iota = iota num_threads

            -- Load the tile 
            let sh_tile = 
                flatten (map (\ tid -> 
                  loop sh_tile for k < e do
                      sh_tile with [tid*e + k] = arr[blkid][tid*e + k]
                ) num_threads_iota)
            
            -- Local sort, bit iterations of 1-bit split
            let psss = replicate num_threads (0,0)
            let _ = map (\ tid ->
                let sh_tile = loop sh_tile for i < b do
                    -- thread local histogram
                    -- make prettier?
                    let elems = sh_tile[tid*e: tid*e+e]
                    let ps0 = reduce (\ps0 x -> 
                        let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                        in ps0 + bit^1
                        ) 0 elems 
                    let ps1 = reduce (\ps1 x -> 
                        let bit = ((x >> (digit*4 + (u32.i64 i))) & 1) 
                        in ps1 + bit
                        ) 0 elems 
                    let psss[tid] = (ps0, ps1)
                    
                    -- scan 
                    let agg = reduce (\ (a0, _) (x, _) -> (0, a0+x)) (0,0) psss
                    let sc = scan (\(e1,e2) (acc1, acc2) -> (e1 + acc1, e2 + acc2)) agg psss
                    let sc_excl = [(0,0)] ++ init sc 
                    
                    -- scatter
                    let idxs = replicate e 0i64                
                    let _ = loop (ps0, ps1) = sc_excl[tid] for x < e do
                              let ele = sh_tile[tid+x]
                              let bit = ((ele >> (digit*4 + (u32.i64 i))) & 1)
                              let idxs[x] = if bit == 0 then i64.u32 ps0 else i64.u32 ps1
                              in (ps0 + bit^1, ps1 + bit)

                    in scatter sh_tile idxs (sh_tile[tid*e : tid*e+e] :> [e]u32)-- MAYBE NEED COPY ?
                in tid
            ) num_threads_iota

            -- blockwide hist
            -- !!!!!SEQ!!!!! fix pls
            let sh_hist = loop sh_hist for i < e*num_threads do
                let ele = sh_tile[i]
                let dig = ele >> (digit * 4) & 0xFF
                let sh_hist[dig] = sh_hist[dig] + 1
                in sh_hist 

            -- write back sorted tile
            cmap (num_threads_iota) (\ tid ->
                loop arr for k < e do
                    arr[blkid] with [tid*e + k] = sh_tile[tid*e + k]
            )
            
            -- writeback histogram
            cmap (iota (2**b)) (\ i -> 
                g_hist[blkid] with [i] = sh_hist[i]
            )
            
            -- what to put here
            in blkid
        )

        in g_hist


