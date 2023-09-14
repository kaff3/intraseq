import "lib/github.com/diku-dk/sorts/radix_sort"
-- ==
-- entry: test_main
-- compiled random input {[10000]u32}
-- compiled random input {[100000]u32}
-- compiled random input {[1000000]u32}
-- compiled random input {[10000000]u32}
entry test_main [n] (inp: [n] u32)  =
    radix_sort i32.num_bits u32.get_bit inp


