let cmap as f = map f as

-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}
let step [num_blocks] [num_threads] [e] (digit : u32) (arr : *[num_blocks][num_threads*e]u32)
          : *[num_blocks][2**4]u32 =
    let b = 4

    let (g_hist, arr) = cmap (iota num_blocks)
        ( \ blkid ->
            let sh_hist = replicate (2**b) 0u32
            -- Load tile
            let sh_tile = arr[blkid]

            let sh_tile = loop sh_tile for i < b do 
                let (p0, p1) = partition (\e -> 
                    if (e >> (digit*4 + (u32.i64 i))) & 1 == 0 then true else false
                ) sh_tile
                in (p0 ++ p1) :> [num_threads*e]u32

            -- blockwide hist
            -- !!!!!SEQ!!!!! fix pls
            let sh_hist = loop sh_hist for i < e*num_threads do
                let ele = sh_tile[i]
                let dig = i64.u32 ((ele >> (digit * 4)) & 0xFF)
                in sh_hist with [dig] = sh_hist[dig] + 1

            in (sh_hist, sh_tile)
        ) |> unzip

        in g_hist


