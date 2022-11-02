-- radix sort: Baseline Version
-- ==
-- entry: sort_i64
-- compiled input {  [65i64, 3i64, 32i64, 10i64, 11i64, 77i64, 5i64, 101i64, 3i64, 35i64] } output {[3i64, 3i64, 5i64, 10i64, 11i64, 32i64, 35i64, 65i64, 77i64, 101i64] }

import "lib/github.com/diku-dk/sorts/radix_sort"

entry sort_u8  = radix_sort_int u8.num_bits u8.get_bit
entry sort_u16 = radix_sort_int u16.num_bits u16.get_bit
entry sort_u32 = radix_sort_int u32.num_bits u32.get_bit
entry sort_u64 = radix_sort_int u64.num_bits u64.get_bit

--echo "[3,4,6,3,6,3]"" | ./b -t /dev/stderr -r 10 > /dev/null
-- let main [n] (arr : [n]i64) : [n]i64 =
--     let sort =  radix_sort_int i64.num_bits i64.get_bit arr
--     in sort
    