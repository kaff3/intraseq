let cmap as f = 
  #[incremental_flattening(only_intra)] 
  #[seq_factor(4)]
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


let step [num_blocks] [num_elems] (digit : u32) (arr : *[num_blocks][num_elems]u32)
          : *[num_blocks][num_elems]u32 =
    -- rankKernel
    let (_, arr_intra) = (cmap arr
        (\row ->
            partition2 (\ele -> 
              if (ele >> digit) & 1 == 0 then true else false
            ) row
        ) |> unzip)
   
    let (_, arr_sorted) = 
        partition2 (\ele -> 
          if (ele >> digit) & 1 == 0 then true else false
        ) (flatten arr_intra)
    in (unflatten arr_sorted)

-- ==
-- entry: main
-- compiled random input { [100000][1024]u32 } auto output
let main [num_blocks] [num_elems] (arr : *[num_blocks][num_elems]u32) 
          : *[num_blocks][num_elems]u32 =
    let num_digits = 32
    in loop arr for i < num_digits do
      step (u32.i32 i) arr


