-- radix sort using 5-bit split
-- based upon https://github.com/diku-dk/sorts/blob/master/lib/github.com/diku-dk/sorts/radix_sort.fut
import "lib/github.com/diku-dk/sorts/radix_sort"

def radix_sort_step_4 [n] 't (xs : [n]t) (get_bit: i32 -> t -> i32) (digit_n : i32) : [n]t =
    let num x =   (get_bit (digit_n+3) x << 3)
                + (get_bit (digit_n+2) x << 2) 
                + (get_bit (digit_n+1) x << 1) 
                + get_bit digit_n x
    let pairwise 
            (a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1,o1,p1) 
            (a2,b2,c2,d2,e2,f2,g2,h2,i2,j2,k2,l2,m2,n2,o2,p2) =
        (a1 + a2, b1 + b2, c1 + c2, d1 + d2, e1 + e2, f1 + f2, g1 + g2, h1 + h2, i1 + i2, j1 + j2, k1 + k2, l1 + l2, m1 + m2, n1 + n2, o1 + o2, p1 + p2) 
    let bins = xs |> map num
    let flags = bins |> map (\x -> match x 
                                case 0  ->(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                                case 1  ->(0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
                                case 2  ->(0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)
                                case 3  ->(0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0)
                                case 4  ->(0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0)
                                case 5  ->(0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0)
                                case 6  ->(0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0)
                                case 7  ->(0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0)
                                case 8  ->(0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0)
                                case 9  ->(0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)
                                case 10 ->(0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0)
                                case 11 ->(0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0)
                                case 12 ->(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)
                                case 13 ->(0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0)
                                case 14 ->(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0)
                                case _ -> (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)
                            )
    let offsets = scan pairwise (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) flags
    let (na,nb,nc,nd,ne,nf,ng,nh,ni,nj,nk,nl,nm,nn,no,_np) = last offsets
    let fun bin (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) = match bin
                            case 0  -> a-1
                            case 1  -> na+b-1
                            case 2  -> na+nb+c-1
                            case 3  -> na+nb+nc+d-1
                            case 4  -> na+nb+nc+nd+e-1
                            case 5  -> na+nb+nc+nd+ne+f-1
                            case 6  -> na+nb+nc+nd+ne+nf+g-1
                            case 7  -> na+nb+nc+nd+ne+nf+ng+h-1
                            case 8  -> na+nb+nc+nd+ne+nf+ng+nh+i-1
                            case 9  -> na+nb+nc+nd+ne+nf+ng+nh+ni+j-1
                            case 10 -> na+nb+nc+nd+ne+nf+ng+nh+ni+nj+k-1
                            case 11 -> na+nb+nc+nd+ne+nf+ng+nh+ni+nj+nk+l-1
                            case 12 -> na+nb+nc+nd+ne+nf+ng+nh+ni+nj+nk+nl+m-1
                            case 13 -> na+nb+nc+nd+ne+nf+ng+nh+ni+nj+nk+nl+nm+n-1
                            case 14 -> na+nb+nc+nd+ne+nf+ng+nh+ni+nj+nk+nl+nm+nn+o-1                            
                            case _ ->  na+nb+nc+nd+ne+nf+ng+nh+ni+nj+nk+nl+nm+nn+no+p-1
    let is = map2 fun bins offsets
    in scatter (copy xs) is xs
    
-- | do the sort
def radix_sort_4 [n] 't (num_bits: i32) (get_bit: i32 -> t -> i32) (xs : [n]t) : [n]t =
    let iters = if n == 0 then 0 else (num_bits+4-1)/4 
    in loop xs for i < iters do radix_sort_step_4 xs get_bit (i*4)

-- ==
-- entry: test_main
-- compiled random input {[1000000]u32}
-- output { false }
entry test_main [n] (inp: [n] u32)  =
    let res_4 = radix_sort_4 i32.num_bits u32.get_bit inp
    let res_2 = radix_sort i32.num_bits u32.get_bit inp
    let trace_4 = map (\x -> trace x) res_4
    let trace_msg = trace "hello friend \n"
    let trace_2 = map (\x -> trace x) res_2
    in any (\(x, y) -> x != y) (zip res_4 res_2)

