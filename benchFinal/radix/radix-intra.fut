let mapSeq as f = 
  #[incremental_flattening(only_intra)] 
  #[seq_factor(4)]
  map f as

let mapOrg as f = 
  #[incremental_flattening(only_intra)] 
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


let stepSeq [num_blocks] [num_elems] (digit : u32) (arr : *[num_blocks][num_elems]u32)
          : *[num_blocks][num_elems]u32 =
    let b = 4
    -- intra block sorting
    let arr_intra = (mapSeq arr
        (\row ->
          let arr = loop row for j < b do
              let (_, arr) = partition2 (\ele -> 
                if (ele >> (digit*4 + (u32.i64 j))) & 1 == 0 then true else false
              ) row
              in arr 
          in arr
        ))
    in arr_intra   

let stepOrg [num_blocks] [num_elems] (digit : u32) (arr : *[num_blocks][num_elems]u32)
          : *[num_blocks][num_elems]u32 =
    let b = 4
    -- intra block sorting
    let arr_intra = (mapOrg arr
        (\row ->
          let arr = loop row for j < b do
              let (_, arr) = partition2 (\ele -> 
                if (ele >> (digit*4 + (u32.i64 j))) & 1 == 0 then true else false
              ) row
              in arr 
          in arr
        ))
    in arr_intra   

-- Intra-group radix sort: performance
-- ==
-- entry: mainSeq 
-- compiled random input { [32768][4096]u32 }
-- compiled random input { [65536][2048]u32 }
-- compiled random input { [131072][1024]u32 } 
-- compiled random input { [262144][512]u32  }
-- compiled random input { [524288][256]u32  }
-- compiled random input { [1048576][128]u32 }

-- Big-Integer Addition: performance
-- ==
-- entry: mainOrg
-- compiled random input { [131072][1024]u32  }
-- compiled random input { [262144][512]u32   }
-- compiled random input { [524288][256]u32   }
-- compiled random input { [1048576][128]u32  }
  
entry mainSeq [num_blocks] [num_elems] (arr : *[num_blocks][num_elems]u32) 
          : *[num_blocks][num_elems]u32 =
    let num_digits = 8
    in loop arr for i < num_digits do
      stepSeq (u32.i32 i) arr

entry mainOrg [num_blocks] [num_elems] (arr : *[num_blocks][num_elems]u32) 
          : *[num_blocks][num_elems]u32 =
    let num_digits = 8
    in loop arr for i < num_digits do
      stepOrg (u32.i32 i) arr

