-----------------------------------------------------------------------------
--- Implementation took heavy inspiration from:
--- [1] Amar Topalovic, Walter Restelli-Nielsen, Kristian Olesen:
---     ``Multiple-precision Integer Arithmetic'', DPP'22 final project,
---     https://futhark-lang.org/student-projects/dpp21-mpint.pdf
-----------------------------------------------------------------------------

let imap2Seq as bs f = 
  #[incremental_flattening(only_intra)]
  #[seq_factor(4)]
  map2 f as bs

let imap2Org as bs f = 
  #[incremental_flattening(only_intra)]
  map2 f as bs
------------------------------------------------------------------------
---- prefix sum (scan) operator to propagate the carry
-- let add_op (ov1 : bool, mx1: bool) (ov2 : bool, mx2: bool) : (bool, bool) =
--   ( (ov1 && mx2) || ov2,    mx1 && mx2 )
------------------------------------------------------------------------

---- prefix sum (scan) operator to propagate the curry:
---- format: last digit set      => overfolow
----         ante-last digit set => one unit away from overflowing   
let badd_op (c1 : u8) (c2: u8) : u8 =
  (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1)
  
let badd [n] (as : [n]u32) (bs : [n]u32) : [n]u32 =
  let (pres, cs) = 
    imap2 as bs
      (\ a b -> let s = a + b 
                let b = u8.bool (s < a)
                let b = b | ((u8.bool (s == u32.highest)) << 1)
                in  (s, b)
      ) |> unzip
  let carries = scan badd_op 2u8 cs
  in  
    imap2 (iota n) pres
        (\ i r -> r + u32.bool (i > 0 && ( (#[unsafe] carries[i-1]) & 1u8 == 1u8)) )

-- Big-Integer Addition: performance
-- ==
-- entry: mainSeq 
-- compiled random input { [32768][4096]u32     [32768][4096]u32 }
-- compiled random input { [65536][2048]u32     [65536][2048]u32 }
-- compiled random input { [131072][1024]u32    [131072][1024]u32 }
-- compiled random input { [262144][512]u32     [262144][512]u32 }
-- compiled random input { [524288][256]u32     [524288][256]u32 }
-- compiled random input { [1048576][128]u32    [1048576][128]u32 }

-- Big-Integer Addition: performance
-- ==
-- entry: mainOrg
-- compiled random input { [131072][1024]u32    [131072][1024]u32 }
-- compiled random input { [262144][512]u32     [262144][512]u32 }
-- compiled random input { [524288][256]u32     [524288][256]u32 }
-- compiled random input { [1048576][128]u32    [1048576][128]u32 }
  
-- computes one batched multiplication: a*b
entry mainSeq [m][n] (ass: [m][n]u32) (bss: [m][n]u32) : [m][n]u32 =
  imap2Seq badd ass bss 

entry mainOrg [m][n] (ass: [m][n]u32) (bss: [m][n]u32) : [m][n]u32 =
  imap2Org badd ass bss 
