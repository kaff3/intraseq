import "lib/github.com/diku-dk/sorts/radix_sort"
-- ==
-- entry: test_main
-- compiled random input {[102400000]u32}
entry test_main [n] (inp: [n] u32)  =
    -- #[incremental_flattening(only_intra)]
    radix_sort i32.num_bits u32.get_bit inp


