-- radix sort: Baseline Version

-- entry: sort_i64
-- compiled input {  [65i64, 3i64, 32i64, 10i64, 11i64, 77i64, 5i64, 101i64, 3i64, 35i64] } output {[3i64, 3i64, 5i64, 10i64, 11i64, 32i64, 35i64, 65i64, 77i64, 101i64] }

import "lib/github.com/diku-dk/sorts/radix_sort"

entry sort_i64 = radix_sort_int i64.num_bits i64.get_bit
--
let main [n]i64 : [n]i64 =
    let sort =  radix_sort_int i64.num_bits i64.get_bit
    in sort

