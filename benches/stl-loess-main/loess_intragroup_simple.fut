
-- nobench input @ experiment_test.in output @ experiment_test.out

module loess_experiment = {
module T = f64
type t = T.t

-- main LOESS procedure
let loess_proc [n] [n_m] (xx: [n]t)           -- indexes of non-nan indexes
                         (yy: [n]t)           -- the corresponding y values
                         (q: i64)             -- span of smoothing
                         (m_fun: i64 -> i64)  -- function that computes m indexes
                         (ww: [n]t)           -- user-defined weights
                         (l_idx: [n_m]i64)    -- index of left starting points
                         (max_dist: [n_m]t)   -- distance between nn bounds for each point
                         (n_nn: i64)          -- number of non-nan values
                         (fit_fun: (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t))
                         (slope_fun: (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t))
                         : ([n_m]t, [n_m]t) =
  let q_slice (arr: [n]t) (l_idx_i: i64) (v: t): [q]t =
    #[unsafe]
    tabulate q (\j -> if j >= n_nn then (T.i64 0) else arr[l_idx_i + j] + v)
  in
  -- [n_m]
  #[incremental_flattening(only_intra)]
  map3 (\i l_idx_i max_dist_i ->
         -- [q]
         -- get polynomial weights (from tri-cube), x, and a
         #[unsafe]
         let xx_slice = q_slice xx l_idx_i 1
         let ww_slice = q_slice ww l_idx_i 0
         let (x, w) =
           map2 (\xx_j ww_j ->
                   let x_j = xx_j - (m_fun i |> T.i64)
                   -- tricube
                   let r = T.abs x_j
                   let tmp1 = r / max_dist_i
                   let tmp2 = 1.0 - tmp1 * tmp1 * tmp1
                   let tmp3 = tmp2 * tmp2 * tmp2
                   -- scale by user-defined weights
                   let tmp4 = tmp3 * ww_j
                   in (x_j, tmp4)
                ) xx_slice ww_slice |> unzip2
         -- then, compute fit and slope based on polynomial degree
         let xw = map2 (*) x w
         let x2w = map2 (*) x xw
         let x3w = map2 (*) x x2w
         let x4w = map2 (*) x x3w

         let a = T.sum w
         let b = T.sum xw
         let c = T.sum x2w
         let d = T.sum x3w
         let e = T.sum x4w

         let det1 = 1 / (a * c - b * b)
         let a11 = c * det1
         let b11 = -b * det1
         let c11 = a * det1

         -- degree 2
         let a12 = e * c - d * d
         let b12 = c * d - e * b
         let c12 = b * d - c * c
         let a2 = c * d - e * b
         let b2 = e * a - c * c
         let c2 = b * c - d * a
         let det = 1 / (a * a12 + b * b12 + c * c12)
         let a12 = a12 * det
         let b12 = b12 * det
         let c12 = c12 * det
         let a2 = a2 * det
         let b2 = b2 * det
         let c2 = c2 * det

         let a0 = 1 / a

         let yy_slice = q_slice yy l_idx_i 0

         let fit =
           map4 (
             \w_j yy_j xw_j x2w_j ->
               fit_fun w_j yy_j xw_j x2w_j a0 a11 a12 b11 b12 c12
           ) w yy_slice xw x2w |> T.sum

         let slope =
           map4 (
             \w_j yy_j xw_j x2w_j ->
               slope_fun w_j yy_j xw_j x2w_j c2 a11 a2 b11 b2 c11
           ) w yy_slice xw x2w |> T.sum
         in (fit, slope)
       ) (iota n_m) l_idx max_dist |> unzip

-- let fit_fun_zero (w_j: t) (yy_j: t) (_: t) (_: t) (a0: t) (_: t) (_: t) (_: t) (_: t) (_: t): t =
--   w_j * a0 * yy_j

-- let fit_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t) (a11: t) (_: t) (b11: t) (_: t) (_: t): t =
--   (w_j * a11 + xw_j * b11) * yy_j

let fit_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (_: t) (_: t) (a12: t) (_: t) (b12: t) (c12: t): t =
  (w_j * a12 + xw_j * b12 + x2w_j * c12) * yy_j


-- let slope_fun_zero (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t) (_: t): t =
--   T.i64 0

-- let slope_fun_one (w_j: t) (yy_j: t) (xw_j: t) (_: t) (_: t) (_: t) (_: t) (b11: t) (_: t) (c11: t): t =
--   (w_j * b11 + xw_j * c11) * yy_j

let slope_fun_two (w_j: t) (yy_j: t) (xw_j: t) (x2w_j: t) (c2: t) (_: t) (a2: t) (_: t) (b2: t) (_: t): t =
  (w_j * a2 + xw_j * b2 + x2w_j * c2) * yy_j


let loess_l [m] [n] [n_m] (xx_l: [m][n]t)        -- time values - should be 1:n unless there are nans
                          (yy_l: [m][n]t)          -- the corresponding y values
                          (degree: i64)            -- polynomial degree 0, 1 or 2
                          (q: i64)                 -- span of smoothing
                          (m_fun: i64 -> i64)      -- function that computes m indexes
                          (ww_l: [m][n]t)          -- user-defined weights
                          (l_idx_l: [m][n_m]i64)   -- index of left starting points
                          (max_dist_l: [m][n_m]t)  -- distance between nn bounds for each point
                          (n_nn_l: [m]i64)         -- number of non-nan values
                          : ([m][n_m]t, [m][n_m]t) =
  let loess_l_fun (fit_fun: (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t))
                  (slope_fun: (t -> t -> t -> t -> t -> t -> t -> t -> t -> t -> t))
                  : ([m][n_m]t, [m][n_m]t) =
    #[incremental_flattening(only_inner)]
    map5 (\xx yy ww l_idx (max_dist, n_nn) ->
            loess_proc xx
                       yy
                       q
                       m_fun
                       ww
                       l_idx
                       max_dist
                       n_nn
                       fit_fun
                       slope_fun
         ) xx_l yy_l ww_l l_idx_l (zip max_dist_l n_nn_l) |> unzip
  in
  -- match degree
  -- case 0 -> loess_l_fun fit_fun_zero slope_fun_zero |> opaque
  -- case 1 -> loess_l_fun fit_fun_one  slope_fun_one  |> opaque
  -- case _ -> loess_l_fun fit_fun_two  slope_fun_two  |> opaque
  loess_l_fun fit_fun_two slope_fun_two
}


entry main [m] [n] [n_m] (xx_l: [m][n]f64)
                         (yy_l: [m][n]f64)
                         (ww_l: [m][n]f64)
                         (l_idx_l: [m][n_m]i64)
                         (max_dist_l: [m][n_m]f64)
                         (n_nn_l: [m]i64)
                         (degree: i64)
                         (q: i64)
                         (jump: i64)
                        : ([m][n_m]f64, [m][n_m]f64) =
  let m_fun (x: i64): i64 = 2 + i64.min (x * jump) (n - 1)
  in
  loess_experiment.loess_l xx_l
                           yy_l
                           degree
                           q
                           m_fun
                           ww_l
                           l_idx_l
                           max_dist_l
                           n_nn_l
