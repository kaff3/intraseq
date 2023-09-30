let cmap as f = map f as

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
  [ne] ++ tail (scan op ne arr) :> [n]t


-- e: the number of elements pr. thread
-- ==
-- entry: step
-- compiled random input {22i64 1i64 1024i64 [22528]u32}
-- compiled input {0u32 [[3,2,1],[5,4,6]]u32}
let step [num_blocks] [num_elems] (e : i64) (digit : u32) (arr : *[num_blocks][num_elems]u32)
          : *[num_blocks][num_elems]u32 =
    let num_threads = num_elems / e
    let b = 4
    let num_blocks_iota = iota num_blocks

    -- rankKernel
    let (g_hist, arr_intra) = cmap (num_blocks_iota)
        (\blkid ->
            let sh_hist = replicate (2**b) 0u32
            -- Load tile
            let sh_tile = arr[blkid]
            
            let sh_tile = loop sh_tile for j < b do 
                let (_, arr) = partition2 (\e -> 
                    if (e >> (digit*4 + (u32.i64 j))) & 1 == 0 then true else false
                ) sh_tile
                in arr
        
            let sh_hist = loop sh_hist for i < num_threads do
                let ele = sh_tile[i]
                let dig = i64.u32 ((ele >> (digit * 4)) & 0xF)
                in sh_hist with [dig] = sh_hist[dig] + 1

            in (sh_hist, sh_tile)
        ) |> unzip
    
    let ghst = transpose g_hist |> flatten

    -- global hist scan
    let ghs = transpose g_hist 
              |> flatten
              |> excl_scan (+) 0 
              |> unflatten
              |> transpose

    -- local hists scan
    let lhs = #[break] map (\blkid -> excl_scan (+) 0 g_hist[blkid]) num_blocks_iota
    
    -- globalScatterKernel
    let idxs = map (\blkid ->
        map (\tid ->
            let dig = i64.u32 ((arr_intra[blkid][tid] >> (digit * 4)) & 0xF) 
            let g_pos = ghs[blkid][dig] 
            let l_pos = u32.i64 tid - lhs[blkid][dig]
            in i64.u32 (g_pos + l_pos)
            ) (iota num_threads)
    ) num_blocks_iota
    let test = #[break] 2
    in scatter (flatten arr :> [num_blocks*num_elems]u32) (flatten idxs :> [num_blocks*num_elems]i64) (flatten arr_intra :> [num_blocks*num_elems]u32)
      |> unflatten


            -- T full_val = d_in[index];
            -- T val = GET_DIGIT(full_val, digit*B, mask);
            
            -- // global_pos: position in global array where this block should place its values with the found digits
            -- // local_pos:  position relative to other values with same digit in this block
            -- unsigned int global_pos = s_histogram_global_scan[val];
            -- unsigned int local_pos = loc_idx - s_histogram[val];  
            -- unsigned int pos = global_pos + local_pos;

let main [num_blocks] [num_elems] (arr : *[num_blocks][num_elems]u32) 
          : *[num_blocks][num_elems]u32 =
    let num_digits = 4
    let thread_elems = 1
    in loop arr for i < num_digits do
      step thread_elems (u32.i32 i) arr  
