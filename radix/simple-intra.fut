let imap as f = map f as

-- ==
-- entry: main
-- random input { 256i64 22i64 [4096][5632]f32} auto output

let main [num_blocks] (b: i64) (m: i64) (arr: [num_blocks][]f32) =
  imap (iota num_blocks)
    (\ blkid ->
        let shmem = replicate (b*m) 0f32
        let shmem =
          loop shmem for k < m do
            let vals = map (\ i -> arr[blkid][k*b + i]) (iota b)
            let iota_arr = map (\ i -> k*b + i) (iota b)
            in  scatter shmem iota_arr vals
        let shmem =
          loop shmem for k < m do
            let shmem[k*b: k*b+b] = scan (+) 0 (shmem[k*b: k*b+b])
            in  shmem
        in shmem
    )
    
-- Autotune with:
--   $ futhark autotune --backend=cuda simple-intra.fut -r 10
-- will create file simple-intra.fut.tuning
-- then you can run
--    $ futhark bench --backend=cuda simple-intra.fut
-- or you can compile and run:
-- 
--    $ futhark dataset -b --i64-bounds=256:256 -g i64 --i64-bounds=22:22 -g i64 -g [4096][5632]f32 > data.in
--    $ futhark cuda simple-intra.fut
--    $ ./simple-intra -t /dev/stderr -r 10 -L -P -n < data.in
