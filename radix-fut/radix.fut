let cmap as f = map f as

-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}
let step [num_blocks] (e : i64) (digit : i32) (num_threads : i64) (arr : [num_blocks][]u32) =
    let b_splits = 4
    let b = 4
    
    let g_hist = replicate (2**b_splits * num_blocks) 0u32
              |> unflatten num_blocks (2**b_splits)
    
    -- rankKernel
    cmap (iota num_blocks)
        ( \ blkid ->
        let sh_hist = replicate (2**b) 0u64
        let sh_tile = replicate (e*num_threads) 0u64
        let num_threads_iota = iota num_threads

        -- Load the tile 
        map (\ tid -> 
            loop sh_tile for k < e do
                let v = arr[blkid][tid*e + k]
                in sh_tile[tid*e + k]
        ) num_threads_iota
        
        -- Local sort, bit iterations of 1-bit split
        map (\ tid ->
            loop sh_tile for i < b do
                let psss = replicate num_threads (0,0)
                    -- thread local histogram
                    let pss = reduce (\x (ps0, ps1) -> 
                        let bit = ((x >> (digit*b + i)) & 1) 
                        in (ps0 + bit^1, ps1 + bit)
                        ) sh_tile[tid*e : (tid+1)*e] (0,0)  
                    let psss[tid] = pss
                    in psss     
                
                -- scan 
                let agg = reduce (\ (a0, _) (x, _) -> (0, a0+x)) (0,0) psss 
                let sc = scan (\(e1,e2) (acc1, acc2) -> (e1 + acc1, e2 + acc2)) agg psss
                let sc_excl = [(0,0)] ++ init sc
                
                -- scatter
                let idxs = replicate e 0                
                loop (ps0, ps1) = sc_excl[tid] for x < e do
                    let idxs[x] = if bit == 0 then ps0 else ps1
              arr      let ele = sh_tile[tid+x]
                    let bit = ((ele >> (digit*b + i)) & 1)
                    in (ps0 + bit^1, ps1 + bit)

                in scatter sh_tile idxs sh_tile[tid*e : (tid+1)*e] -- MAYBE NEED COPY ??       
        ) num_threads_iota

        -- blockwide hist
        -- !!!!!SEQ!!!!! fix pls
        loop sh_hist for i < e*num_threads do
            let ele = sh_tile[i]
            let digit = ele >> (digit * b) & 0xFF
            let sh_hist[digit] = sh_hist[digit] + 1
            in sh_hist 

        -- write back sorted tile
        cmap (num_threads_iota) (\ tid ->
            -- HACK REWRITE
            loop let _ = 0 for k < e do
                let arr[blkid][tid*e + k] = sh_tile[tid*e + k]
                in 0
        )

        -- writeback histogram
        cmap (iota (2**b)) (\ i -> 
            -- global hist
            g_hist[blkid][i] = sh_hist[i]
        )
    )


