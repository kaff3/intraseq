let imap  as f = map f as
let imap2 as bs f = map2 f as bs
let imap2Intra as bs f = #[incremental_flattening(only_intra)] map2 f as bs


-----------------------------------------------------------------------------
--- Implementation took inspiration from:
--- [1] Amar Topalovic, Walter Restelli-Nielsen, Kristian Olesen:
---     ``Multiple-precision Integer Arithmetic'', DPP'22 final project,
---     https://futhark-lang.org/student-projects/dpp21-mpint.pdf
-----------------------------------------------------------------------------

type cT         = u32      --u8
let  cTfromBool = u32.bool --u8.bool
let  two_cT     = 2u32     --2u8

--type cT         = u8
--let  cTfromBool = u8.bool
--let  two_cT     = 2u8


------------------------------------------------------------------------
---- prefix sum (scan) operator to propagate the carry
-- let add_op (ov1 : bool, mx1: bool) (ov2 : bool, mx2: bool) : (bool, bool) =
--   ( (ov1 && mx2) || ov2,    mx1 && mx2 )
------------------------------------------------------------------------

---- prefix sum (scan) operator to propagate the curry:
---- format: last digit set      => overfolow
----         ante-last digit set => one unit away from overflowing   
-- let badd_op (c1 : u8) (c2: u8) : u8 =
let carryOp (c1: cT) (c2: cT) =
  (c1 & c2 & 2) | (( (c1 & (c2 >> 1)) | c2) & 1)
  
let carrySegOp (c1: cT) (c2: cT) =
    if (c2 & 4) != 0 then c2
    else let res = ( (c1 & (c2 >> 1)) | c2 ) & 1
         let res = res | (c1 & c2  & 2)
         in  ( res | ( (c1 | c2) & 4 ) )

let carryOpNE: cT = two_cT
  
let addPairwise (m: i32) (ash: []u64) (bsh: []u64) (tid: i32) (i: i32) : (u64, cT)=
  let ind = tid * 4 + i
  let (a,b) = ( #[unsafe] ash[ind], #[unsafe] bsh[ind] )
  let r = a + b
  let c = cTfromBool (r < a)
  let c = c | ( (cTfromBool (r == u64.highest)) << 1 )
  let c = c | ( (cTfromBool ( (ind % m) == 0 )) << 2 )
  in  (r, c)
  
let badd [ipb][n] (as : [ipb*(4*n)]u64) (bs : [ipb*(4*n)]u64) : [ipb*(4*n)]u64 =
  let nn = i32.i64 n
  let g = i64.i32 ((i32.i64 ipb) * nn)

  let cp2sh (i : i32) = #[unsafe]
        let g = i32.i64 g in
        ( ( as[i], as[g + i], as[2*g + i], as[3*g + i] )
        , ( bs[i], bs[g + i], bs[2*g + i], bs[3*g + i] ) )

  let ( ass, bss ) = iota g |> map i32.i64
                  |> map cp2sh  |> unzip
  let (a1s, a2s, a3s, a4s) = unzip4 ass
  let (b1s, b2s, b3s, b4s) = unzip4 bss
  let ash = a1s ++ a2s ++ a3s ++ a4s
  let bsh = b1s ++ b2s ++ b3s ++ b4s
  let ash = ash |> opaque
  let bsh = bsh |> opaque

--  let cp2sh (as: []u64) (bs: []u64) (offset: i32) (tid: i32)  =
--    ( #[unsafe] as[offset + tid], #[unsafe] bs[offset+tid] )
--
--  let ash = replicate (ipb*(4*n)) 0
--  let bsh = replicate (ipb*(4*n)) 0
--  let offset = 0i64
--  let (ash, bsh, _) =
--    loop (ash, bsh, offset) for _i < 4 do
--        let (a_slice, b_slice) = 
--            iota g |> map i32.i64 |> map (cp2sh as bs (i32.i64 offset)) |> unzip
--        let next_offset = offset + g
--        let ash[offset: next_offset] = a_slice
--        let bsh[offset: next_offset] = b_slice
--        in  (ash, bsh, next_offset)

  let seqred4 (tid: i32) =
    loop (accum) = (carryOpNE) for i < 4 do
        let (_, c) = addPairwise (4 * nn) ash bsh tid i
        in  carrySegOp accum c
  
  let seqscan1 (tid: i32) (i: i32) (carry: cT) =
    let (r0, c0) = addPairwise (4 * nn) ash bsh tid i
    let r0 = r0 + u64.bool ( ( (c0 & 4) == 0 ) && ( (carry & 1) == 1 ) ) 
    in  (r0, carrySegOp carry c0)

  let seqscan4 (carries: [g]cT) (tid: i32) =
    let carry = if tid == 0 then carryOpNE else #[unsafe] carries[tid-1] 
    let (r0, carry) = seqscan1 tid 0 carry
    let (r1, carry) = seqscan1 tid 1 carry
    let (r2, carry) = seqscan1 tid 2 carry
    let (r3, _)     = seqscan1 tid 3 carry
    in  (r0,r1,r2,r3)  

  let carries = iota g 
             |> map i32.i64
             |> map seqred4
             |> scan carrySegOp carryOpNE 

  let (rs0, rs1, rs2, rs3) = iota g |> map i32.i64 
                          |> map (seqscan4 carries)
                          |> unzip4
  let rs = rs0 ++ rs1 ++ rs2 ++ rs3
  let rs = (rs :> [ipb*(4*n)]u64) |> opaque
  in  rs
--  let rs4 = iota g |> map i32.i64 |> map (seqscan4 carries)
--  in  flatten rs4 :> [ipb*(4*n)]u64

-- Big-Integer Addition: performance
-- ==
-- entry: oneAddition 
-- compiled random input { 256i64 [65536][1][1024]u64  [65536][1][1024]u64 }
-- compiled random input { 128i64 [131072][1][512]u64  [131072][1][512]u64 }
-- compiled random input { 64i64  [262144][1][256]u64  [262144][1][256]u64 }
-- compiled random input { 32i64  [262144][2][128]u64  [262144][2][128]u64 }
-- compiled random input { 16i64  [262144][4][64]u64   [262144][4][64]u64 }
-- compiled random input {  8i64  [262144][8][32]u64   [262144][8][32]u64 }
-- compiled random input {  4i64  [262144][16][16]u64  [262144][16][16]u64 }


-- compiled random input { [1024][4][128]u64  [1024][4][128]u64 }

-- computes one batched multiplication: a + b
entry oneAddition [m][ipb] (n4: i64) (ass0: [m][ipb][4*n4]u64) (bss0: [m][ipb][4*n4]u64) : [m][ipb][4*n4]u64 = #[unsafe]
   let ass = (map flatten ass0) 
   let bss = (map flatten bss0) 
   let rss = imap2Intra ass bss badd |> map unflatten
   in  rss
  
--
-- ==
-- entry: oneAddition128
-- compiled random input { [262144][2][128]u64  [262144][2][128]u64 }

entry oneAddition128 [m] (ass0: [m][2][128]u64) (bss0: [m][2][128]u64) : [][][]u64 = #[unsafe]
   let ass = (map flatten ass0) :> [m][2*(4*32)]u64
   let bss = (map flatten bss0) :> [m][2*(4*32)]u64
   let rss = imap2Intra ass bss badd |> map unflatten
   in  rss


-- computes `6*a + 10*b` as  `b + 3 * ( 2*(a+b) + b )` using 6 additions 
-- entry manyAdditions [m][n] (as: [m][n]u32) (bs: [m][n]u32) : [m][n]u32 =
--   imap2Intra as bs 
--     (\ a b ->
--         -- let (a, b) = (glb2shmem a, glb2shmem  b)
--         let apb    = badd a     b     -- a + b
--         let a2pb2  = badd apb   apb   -- 2 * (a + b) = 2a + 2b
--         let a2pb3  = badd a2pb2 b     -- 2 * (a + b) + b = 2a + 3b
--         let a4pb6  = badd a2pb3 a2pb3 -- 2 * (2a + 3b) = 4a + 6b
--         let a6pb9  = badd a4pb6 a2pb3 -- (4a + 6b) + (2a + 3b) = 6a + 9b
--         let result = badd a6pb9 b     -- 6a + 10b
--         in  result
--     )
