-- radix sort: Baseline Version
-- ==
-- entry: sort_i64
-- compiled input {  [65i64, 3i64, 32i64, 10i64, 11i64, 77i64, 5i64, 101i64, 3i64, 35i64] } output {[3i64, 3i64, 5i64, 10i64, 11i64, 32i64, 35i64, 65i64, 77i64, 101i64] }

import "lib/github.com/diku-dk/sorts/radix_sort"

entry sort_i64 = radix_sort_int i64.num_bits i64.get_bit

--echo "[3,4,6,3,6,3]"" | ./b -t /dev/stderr -r 10 > /dev/null
--futhark dataset --i64-bounds=0:9999 -g [1000000]i64 | ./baseline -t /dev/stderr > /dev/null
let main [n] (arr : [n]i64) : [n]i64 =
    let sort =  radix_sort_int i64.num_bits i64.get_bit arr
    in sort
    
-- One may run, for example, with:
-- $ futhark dataset --i64-bounds=0:9999 -g [1000000]i64 --f32-bounds=-7.0:7.0 -g [1000000]f32 --i64-bounds=100:100 -g [10000]i64 --f32-bounds=-10.0:10.0 -g [10000]f32 | ./spMVmult-seq -t /dev/stderr > /dev/null
--let main [n] [m] 
--         (mat_inds : [n]i64) (mat_vals : [n]f32) 
--         (shp : [m]i64) (vct : []f32) : [m]f32 =
--  spMatVctMult (zip mat_inds mat_vals) shp vct