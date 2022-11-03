import "lib/github.com/diku-dk/sorts/radix_sort"

-- ==
-- entry: sort_u8
-- compiled random input { [1000]u8 } auto output
-- compiled random input { [10000]u8 } auto output
-- compiled random input { [100000]u8 } auto output
-- compiled random input { [1000000]u8 } auto output
-- compiled random input { [10000000]u8 } auto output
-- compiled random input { [100000000]u8 } auto output
entry sort_u8  = radix_sort_int u8.num_bits u8.get_bit
-- ==
-- entry: sort_u16
-- compiled random input { [1000]u16 } auto output
-- compiled random input { [10000]u16 } auto output
-- compiled random input { [100000]u16 } auto output
-- compiled random input { [1000000]u16 } auto output
-- compiled random input { [10000000]u16 } auto output
-- compiled random input { [100000000]u16 } auto output
entry sort_u16 = radix_sort_int u16.num_bits u16.get_bit
-- ==
-- entry: sort_u32
-- compiled random input { [1000]u32 } auto output
-- compiled random input { [10000]u32 } auto output
-- compiled random input { [100000]u32 } auto output
-- compiled random input { [1000000]u32 } auto output
-- compiled random input { [10000000]u32 } auto output
-- compiled random input { [100000000]u32 } auto output
entry sort_u32 = radix_sort_int u32.num_bits u32.get_bit
-- ==
-- entry: sort_u64
-- compiled random input { [1000]u64 } auto output
-- compiled random input { [10000]u64 } auto output
-- compiled random input { [100000]u64 } auto output
-- compiled random input { [1000000]u64 } auto output
-- compiled random input { [10000000]u64 } auto output
-- compiled random input { [100000000]u64 } auto output
entry sort_u64 = radix_sort_int u64.num_bits u64.get_bit
