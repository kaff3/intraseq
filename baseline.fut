-- radix sort: Baseline Version
-- ==
-- compiled input { 30i64 } output { [2i64, 3i64, 5i64, 7i64, 11i64, 13i64, 17i64, 19i64, 23i64, 29i64] }

import  "lib/github.com/diku-dk/sorts"

-- Run with $echo "1000000" | ./primes-naive -t /dev/stderr -r 10 > /dev/null
let main (n : i64) : []i64 =
