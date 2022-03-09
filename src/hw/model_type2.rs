use crate::hw::node::*;
use crate::stats::norm_cdf;
use crate::ts::TermStructure;

fn fswap(t: f64, tenor: f64, ts: TermStructure) -> f64 {
    let mut denom = 0.0;
    let mut s = t + 0.25;
    while s <= t+tenor {
        denom += ts.p[(12.0*s) as usize];
        s += 0.25;
    }
    (ts.p[(12.0*t) as usize]-ts.p[(12.0*(t+tenor)) as usize])/denom/0.25
}

fn pswaption_black(mat: f64, tenor: f64, black_vol: f64, ts: TermStructure) -> f64 {
    let term1 = ts.p[(12.0*mat) as usize] - ts.p[(12.0*(mat+tenor)) as usize];
    let d1 = 0.5*black_vol*f64::sqrt(mat);
    let cum_prob = norm_cdf(d1);
    return term1*(2.0*cum_prob-1.0);
}

fn gd<F: Fn([f64; 8]) -> (f64, [f64; 8])>(step: F, p0: [f64; 8], lr: f64, tol: f64) -> ([f64; 8], f64, f64) {
    let mut p = p0;
    let mut err;
    let mut grad_norm;
    
    loop {
        let (err0, grad) = step(p);
        err = err0;
        
        // Update
        p[0] -= grad[0]*lr;
        p[0] = p[0].max(0.0001);
        p[1] -= grad[1]*lr;
        p[1] = p[1].max(0.0001);
        p[2] -= grad[2]*lr;
        p[3] -= grad[3]*lr;
        p[4] -= grad[4]*lr;
        p[5] -= grad[5]*lr;
        p[6] -= grad[6]*lr;
        p[7] -= grad[7]*lr;
        
        // Logging
        grad_norm = (grad[2]*grad[2] + grad[3]*grad[3] + grad[4]*grad[4] + grad[5]*grad[5] + grad[6]*grad[6] + grad[7]*grad[7]).sqrt();
        // println!("Result : {:?}", (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], err, grad_norm));

        // Exit
        if grad_norm < tol {
            break;
        }
    }
    
    (p, err, grad_norm)
}

pub fn learning(ts: TermStructure, swaption_vol_mkt: [f64; 36], p0: [f64; 8], lr: f64, tol: f64) -> ([f64; 8], f64, f64) {

    let mut pswaption_mkt = [[0.0; 6]; 6];
    let mat_tenor = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    for i in 0..6 {
        for j in 0..6 {
            pswaption_mkt[i][j] = pswaption_black(mat_tenor[i], mat_tenor[j], swaption_vol_mkt[6*i+j], ts);
        }
    }

    let step = move |p: [f64; 8]| {

        let mut mat: f64;
        
        let alpha10 = p[0];
        let alpha20 = p[1];
        let sigma1 = p[2];
        let sigma2 = p[3];
        let sigma3 = p[4];
        let sigma5 = p[5];
        let sigma7 = p[6];
        let sigma10 = p[7];

        // 1. Modeling
        // 1.1. Option Maturity 1
        mat = 1.0;
        let fswap_m1_t1 = fswap(mat, 1.0, ts);
        let fswap_m1_t2 = fswap(mat, 2.0, ts);
        let fswap_m1_t3 = fswap(mat, 3.0, ts);
        let fswap_m1_t5 = fswap(mat, 5.0, ts);
        let fswap_m1_t7 = fswap(mat, 7.0, ts);
        let fswap_m1_t10 = fswap(mat, 10.0, ts);

        // 1.1.1. Layer 1
        let mut l1_m1_vr = Vr::new(mat);
        let mut l1_m1_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m1_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.1.2. Layer 2
        let mut l2_m1_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m1_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m1_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.1.3. Layer 3
        let mut l3_m1_t1_rstar = RstarT1::new(fswap_m1_t1);
        let mut l3_m1_t2_rstar = RstarT2::new(fswap_m1_t2);
        let mut l3_m1_t3_rstar = RstarT3::new(fswap_m1_t3);
        let mut l3_m1_t5_rstar = RstarT5::new(fswap_m1_t5);
        let mut l3_m1_t7_rstar = RstarT7::new(fswap_m1_t7);
        let mut l3_m1_t10_rstar = RstarT10::new(fswap_m1_t10);
        
        // 1.1.4. Layer 4
        let mut l4_m1_t1_pswaption = PSwaptionT1::new(mat, fswap_m1_t1, ts);
        let mut l4_m1_t2_pswaption = PSwaptionT2::new(mat, fswap_m1_t2, ts);
        let mut l4_m1_t3_pswaption = PSwaptionT3::new(mat, fswap_m1_t3, ts);
        let mut l4_m1_t5_pswaption = PSwaptionT5::new(mat, fswap_m1_t5, ts);
        let mut l4_m1_t7_pswaption = PSwaptionT7::new(mat, fswap_m1_t7, ts);
        let mut l4_m1_t10_pswaption = PSwaptionT10::new(mat, fswap_m1_t10, ts);
        

        // 1.2. Option Maturity 2
        mat = 2.0;
        let fswap_m2_t1 = fswap(mat, 1.0, ts);
        let fswap_m2_t2 = fswap(mat, 2.0, ts);
        let fswap_m2_t3 = fswap(mat, 3.0, ts);
        let fswap_m2_t5 = fswap(mat, 5.0, ts);
        let fswap_m2_t7 = fswap(mat, 7.0, ts);
        let fswap_m2_t10 = fswap(mat, 10.0, ts);

        // 1.2.1. Layer 1
        let mut l1_m2_vr = Vr::new(mat);
        let mut l1_m2_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m2_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.2.2. Layer 2
        let mut l2_m2_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m2_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m2_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.2.3. Layer 3
        let mut l3_m2_t1_rstar = RstarT1::new(fswap_m2_t1);
        let mut l3_m2_t2_rstar = RstarT2::new(fswap_m2_t2);
        let mut l3_m2_t3_rstar = RstarT3::new(fswap_m2_t3);
        let mut l3_m2_t5_rstar = RstarT5::new(fswap_m2_t5);
        let mut l3_m2_t7_rstar = RstarT7::new(fswap_m2_t7);
        let mut l3_m2_t10_rstar = RstarT10::new(fswap_m2_t10);
        
        // 1.2.4. Layer 4
        let mut l4_m2_t1_pswaption = PSwaptionT1::new(mat, fswap_m2_t1, ts);
        let mut l4_m2_t2_pswaption = PSwaptionT2::new(mat, fswap_m2_t2, ts);
        let mut l4_m2_t3_pswaption = PSwaptionT3::new(mat, fswap_m2_t3, ts);
        let mut l4_m2_t5_pswaption = PSwaptionT5::new(mat, fswap_m2_t5, ts);
        let mut l4_m2_t7_pswaption = PSwaptionT7::new(mat, fswap_m2_t7, ts);
        let mut l4_m2_t10_pswaption = PSwaptionT10::new(mat, fswap_m2_t10, ts);


        // 1.3. Option Maturity 3
        mat = 3.0;
        let fswap_m3_t1 = fswap(mat, 1.0, ts);
        let fswap_m3_t2 = fswap(mat, 2.0, ts);
        let fswap_m3_t3 = fswap(mat, 3.0, ts);
        let fswap_m3_t5 = fswap(mat, 5.0, ts);
        let fswap_m3_t7 = fswap(mat, 7.0, ts);
        let fswap_m3_t10 = fswap(mat, 10.0, ts);

        // 1.3.1. Layer 1
        let mut l1_m3_vr = Vr::new(mat);
        let mut l1_m3_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m3_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.3.2. Layer 2
        let mut l2_m3_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m3_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m3_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.3.3. Layer 3
        let mut l3_m3_t1_rstar = RstarT1::new(fswap_m3_t1);
        let mut l3_m3_t2_rstar = RstarT2::new(fswap_m3_t2);
        let mut l3_m3_t3_rstar = RstarT3::new(fswap_m3_t3);
        let mut l3_m3_t5_rstar = RstarT5::new(fswap_m3_t5);
        let mut l3_m3_t7_rstar = RstarT7::new(fswap_m3_t7);
        let mut l3_m3_t10_rstar = RstarT10::new(fswap_m3_t10);
        
        // 1.3.4. Layer 4
        let mut l4_m3_t1_pswaption = PSwaptionT1::new(mat, fswap_m3_t1, ts);
        let mut l4_m3_t2_pswaption = PSwaptionT2::new(mat, fswap_m3_t2, ts);
        let mut l4_m3_t3_pswaption = PSwaptionT3::new(mat, fswap_m3_t3, ts);
        let mut l4_m3_t5_pswaption = PSwaptionT5::new(mat, fswap_m3_t5, ts);
        let mut l4_m3_t7_pswaption = PSwaptionT7::new(mat, fswap_m3_t7, ts);
        let mut l4_m3_t10_pswaption = PSwaptionT10::new(mat, fswap_m3_t10, ts);

        // 1.4. Option Maturity 5
        mat = 5.0;
        let fswap_m5_t1 = fswap(mat, 1.0, ts);
        let fswap_m5_t2 = fswap(mat, 2.0, ts);
        let fswap_m5_t3 = fswap(mat, 3.0, ts);
        let fswap_m5_t5 = fswap(mat, 5.0, ts);
        let fswap_m5_t7 = fswap(mat, 7.0, ts);
        let fswap_m5_t10 = fswap(mat, 10.0, ts);

        // 1.4.1. Layer 1
        let mut l1_m5_vr = Vr::new(mat);
        let mut l1_m5_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m5_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.4.2. Layer 2
        let mut l2_m5_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m5_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m5_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.4.3. Layer 3
        let mut l3_m5_t1_rstar = RstarT1::new(fswap_m5_t1);
        let mut l3_m5_t2_rstar = RstarT2::new(fswap_m5_t2);
        let mut l3_m5_t3_rstar = RstarT3::new(fswap_m5_t3);
        let mut l3_m5_t5_rstar = RstarT5::new(fswap_m5_t5);
        let mut l3_m5_t7_rstar = RstarT7::new(fswap_m5_t7);
        let mut l3_m5_t10_rstar = RstarT10::new(fswap_m5_t10);
        
        // 1.4.4. Layer 4
        let mut l4_m5_t1_pswaption = PSwaptionT1::new(mat, fswap_m5_t1, ts);
        let mut l4_m5_t2_pswaption = PSwaptionT2::new(mat, fswap_m5_t2, ts);
        let mut l4_m5_t3_pswaption = PSwaptionT3::new(mat, fswap_m5_t3, ts);
        let mut l4_m5_t5_pswaption = PSwaptionT5::new(mat, fswap_m5_t5, ts);
        let mut l4_m5_t7_pswaption = PSwaptionT7::new(mat, fswap_m5_t7, ts);
        let mut l4_m5_t10_pswaption = PSwaptionT10::new(mat, fswap_m5_t10, ts);


        // 1.5. Option Maturity 7
        mat = 7.0;
        let fswap_m7_t1 = fswap(mat, 1.0, ts);
        let fswap_m7_t2 = fswap(mat, 2.0, ts);
        let fswap_m7_t3 = fswap(mat, 3.0, ts);
        let fswap_m7_t5 = fswap(mat, 5.0, ts);
        let fswap_m7_t7 = fswap(mat, 7.0, ts);
        let fswap_m7_t10 = fswap(mat, 10.0, ts);

        // 1.5.1. Layer 1
        let mut l1_m7_vr = Vr::new(mat);
        let mut l1_m7_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m7_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.5.2. Layer 2
        let mut l2_m7_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m7_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m7_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.5.3. Layer 3
        let mut l3_m7_t1_rstar = RstarT1::new(fswap_m7_t1);
        let mut l3_m7_t2_rstar = RstarT2::new(fswap_m7_t2);
        let mut l3_m7_t3_rstar = RstarT3::new(fswap_m7_t3);
        let mut l3_m7_t5_rstar = RstarT5::new(fswap_m7_t5);
        let mut l3_m7_t7_rstar = RstarT7::new(fswap_m7_t7);
        let mut l3_m7_t10_rstar = RstarT10::new(fswap_m7_t10);
        
        // 1.5.4. Layer 4
        let mut l4_m7_t1_pswaption = PSwaptionT1::new(mat, fswap_m7_t1, ts);
        let mut l4_m7_t2_pswaption = PSwaptionT2::new(mat, fswap_m7_t2, ts);
        let mut l4_m7_t3_pswaption = PSwaptionT3::new(mat, fswap_m7_t3, ts);
        let mut l4_m7_t5_pswaption = PSwaptionT5::new(mat, fswap_m7_t5, ts);
        let mut l4_m7_t7_pswaption = PSwaptionT7::new(mat, fswap_m7_t7, ts);
        let mut l4_m7_t10_pswaption = PSwaptionT10::new(mat, fswap_m7_t10, ts);


        // 1.6. Option Maturity 10
        mat = 10.0;
        let fswap_m10_t1 = fswap(mat, 1.0, ts);
        let fswap_m10_t2 = fswap(mat, 2.0, ts);
        let fswap_m10_t3 = fswap(mat, 3.0, ts);
        let fswap_m10_t5 = fswap(mat, 5.0, ts);
        let fswap_m10_t7 = fswap(mat, 7.0, ts);
        let fswap_m10_t10 = fswap(mat, 10.0, ts);

        // 1.6.1. Layer 1
        let mut l1_m10_vr = Vr::new(mat);
        let mut l1_m10_b = [B2::new(0.0, 0.0); 40];
        for i in 0..40 {
            l1_m10_b[i] = B2::new(mat, mat+0.25+0.25*(i as f64));
        }
        
        // 1.6.2. Layer 2
        let mut l2_m10_a = [A::new(0.0, 0.0, ts); 40];
        let mut l2_m10_vp = [Vp::new(); 40];
        for i in 0..40 {
            l2_m10_a[i] = A::new(mat, mat+0.25+0.25*(i as f64), ts);
        }
        
        // 1.6.3. Layer 3
        let mut l3_m10_t1_rstar = RstarT1::new(fswap_m10_t1);
        let mut l3_m10_t2_rstar = RstarT2::new(fswap_m10_t2);
        let mut l3_m10_t3_rstar = RstarT3::new(fswap_m10_t3);
        let mut l3_m10_t5_rstar = RstarT5::new(fswap_m10_t5);
        let mut l3_m10_t7_rstar = RstarT7::new(fswap_m10_t7);
        let mut l3_m10_t10_rstar = RstarT10::new(fswap_m10_t10);
        
        // 1.6.4. Layer 4
        let mut l4_m10_t1_pswaption = PSwaptionT1::new(mat, fswap_m10_t1, ts);
        let mut l4_m10_t2_pswaption = PSwaptionT2::new(mat, fswap_m10_t2, ts);
        let mut l4_m10_t3_pswaption = PSwaptionT3::new(mat, fswap_m10_t3, ts);
        let mut l4_m10_t5_pswaption = PSwaptionT5::new(mat, fswap_m10_t5, ts);
        let mut l4_m10_t7_pswaption = PSwaptionT7::new(mat, fswap_m10_t7, ts);
        let mut l4_m10_t10_pswaption = PSwaptionT10::new(mat, fswap_m10_t10, ts);

        // 1.7. Layer 5
        let mut l5_mrse = MRSE::new(pswaption_mkt);
        // let mut l5_mrae = MRAE::new(pswaption_mkt);



        // 2. Feedforward       
        // 2.1. Option Maturity 1
        // 2.1.1. Layer 1
        let mut m1_b = [0.0; 40];
        for i in 0..40 {
            m1_b[i] = l1_m1_b[i].forward(alpha10, alpha20);
        }
        let m1_vr = l1_m1_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);
        
        // 2.1.2. Layer 2
        let mut m1_a = [0.0; 40];
        let mut m1_vp = [0.0; 40];
        for i in 0..40 {
            m1_a[i] = l2_m1_a[i].forward(m1_b[i], m1_vr);
            m1_vp[i] = l2_m1_vp[i].forward(m1_b[i], m1_vr);
        }
        
        // 2.1.3. Layer 3
        let m1_t1_rstar = l3_m1_t1_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3]],
        );
        let m1_t2_rstar = l3_m1_t2_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7]],
        );
        let m1_t3_rstar = l3_m1_t3_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11]],
        );
        let m1_t5_rstar = l3_m1_t5_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19]],
        );
        let m1_t7_rstar = l3_m1_t7_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19], m1_a[20], m1_a[21], m1_a[22], m1_a[23], m1_a[24], m1_a[25], m1_a[26], m1_a[27]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19], m1_b[20], m1_b[21], m1_b[22], m1_b[23], m1_b[24], m1_b[25], m1_b[26], m1_b[27]],
        );
        let m1_t10_rstar = l3_m1_t10_rstar.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19], m1_a[20], m1_a[21], m1_a[22], m1_a[23], m1_a[24], m1_a[25], m1_a[26], m1_a[27], m1_a[28], m1_a[29], m1_a[30], m1_a[31], m1_a[32], m1_a[33], m1_a[34], m1_a[35], m1_a[36], m1_a[37], m1_a[38], m1_a[39]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19], m1_b[20], m1_b[21], m1_b[22], m1_b[23], m1_b[24], m1_b[25], m1_b[26], m1_b[27], m1_b[28], m1_b[29], m1_b[30], m1_b[31], m1_b[32], m1_b[33], m1_b[34], m1_b[35], m1_b[36], m1_b[37], m1_b[38], m1_b[39]],
        );
        
        // 2.1.4. Layer 4
        let m1_t1_pswaption = l4_m1_t1_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3]],
            m1_t1_rstar,
        );
        let m1_t2_pswaption = l4_m1_t2_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3], m1_vp[4], m1_vp[5], m1_vp[6], m1_vp[7]],
            m1_t2_rstar,
        );
        let m1_t3_pswaption = l4_m1_t3_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3], m1_vp[4], m1_vp[5], m1_vp[6], m1_vp[7], m1_vp[8], m1_vp[9], m1_vp[10], m1_vp[11]],
            m1_t3_rstar,
        );
        let m1_t5_pswaption = l4_m1_t5_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3], m1_vp[4], m1_vp[5], m1_vp[6], m1_vp[7], m1_vp[8], m1_vp[9], m1_vp[10], m1_vp[11], m1_vp[12], m1_vp[13], m1_vp[14], m1_vp[15], m1_vp[16], m1_vp[17], m1_vp[18], m1_vp[19]],
            m1_t5_rstar,
        );
        let m1_t7_pswaption = l4_m1_t7_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19], m1_a[20], m1_a[21], m1_a[22], m1_a[23], m1_a[24], m1_a[25], m1_a[26], m1_a[27]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19], m1_b[20], m1_b[21], m1_b[22], m1_b[23], m1_b[24], m1_b[25], m1_b[26], m1_b[27]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3], m1_vp[4], m1_vp[5], m1_vp[6], m1_vp[7], m1_vp[8], m1_vp[9], m1_vp[10], m1_vp[11], m1_vp[12], m1_vp[13], m1_vp[14], m1_vp[15], m1_vp[16], m1_vp[17], m1_vp[18], m1_vp[19], m1_vp[20], m1_vp[21], m1_vp[22], m1_vp[23], m1_vp[24], m1_vp[25], m1_vp[26], m1_vp[27]],
            m1_t7_rstar,
        );
        let m1_t10_pswaption = l4_m1_t10_pswaption.forward(
            [m1_a[0], m1_a[1], m1_a[2], m1_a[3], m1_a[4], m1_a[5], m1_a[6], m1_a[7], m1_a[8], m1_a[9], m1_a[10], m1_a[11], m1_a[12], m1_a[13], m1_a[14], m1_a[15], m1_a[16], m1_a[17], m1_a[18], m1_a[19], m1_a[20], m1_a[21], m1_a[22], m1_a[23], m1_a[24], m1_a[25], m1_a[26], m1_a[27], m1_a[28], m1_a[29], m1_a[30], m1_a[31], m1_a[32], m1_a[33], m1_a[34], m1_a[35], m1_a[36], m1_a[37], m1_a[38], m1_a[39]],
            [m1_b[0], m1_b[1], m1_b[2], m1_b[3], m1_b[4], m1_b[5], m1_b[6], m1_b[7], m1_b[8], m1_b[9], m1_b[10], m1_b[11], m1_b[12], m1_b[13], m1_b[14], m1_b[15], m1_b[16], m1_b[17], m1_b[18], m1_b[19], m1_b[20], m1_b[21], m1_b[22], m1_b[23], m1_b[24], m1_b[25], m1_b[26], m1_b[27], m1_b[28], m1_b[29], m1_b[30], m1_b[31], m1_b[32], m1_b[33], m1_b[34], m1_b[35], m1_b[36], m1_b[37], m1_b[38], m1_b[39]],
            [m1_vp[0], m1_vp[1], m1_vp[2], m1_vp[3], m1_vp[4], m1_vp[5], m1_vp[6], m1_vp[7], m1_vp[8], m1_vp[9], m1_vp[10], m1_vp[11], m1_vp[12], m1_vp[13], m1_vp[14], m1_vp[15], m1_vp[16], m1_vp[17], m1_vp[18], m1_vp[19], m1_vp[20], m1_vp[21], m1_vp[22], m1_vp[23], m1_vp[24], m1_vp[25], m1_vp[26], m1_vp[27], m1_vp[28], m1_vp[29], m1_vp[30], m1_vp[31], m1_vp[32], m1_vp[33], m1_vp[34], m1_vp[35], m1_vp[36], m1_vp[37], m1_vp[38], m1_vp[39]],
            m1_t10_rstar,
        );

        // 2.2. Option Maturity 2
        // 2.2.1. Layer 1
        let mut m2_b = [0.0; 40];
        for i in 0..40 {
            m2_b[i] = l1_m2_b[i].forward(alpha10, alpha20);
        }
        let m2_vr = l1_m2_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);

        // 2.2.2. Layer 2
        let mut m2_a = [0.0; 40];
        let mut m2_vp = [0.0; 40];
        for i in 0..40 {
            m2_a[i] = l2_m2_a[i].forward(m2_b[i], m2_vr);
            m2_vp[i] = l2_m2_vp[i].forward(m2_b[i], m2_vr);
        }

        // 2.2.3. Layer 3
        let m2_t1_rstar = l3_m2_t1_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3]],
        );
        let m2_t2_rstar = l3_m2_t2_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7]],
        );
        let m2_t3_rstar = l3_m2_t3_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11]],
        );
        let m2_t5_rstar = l3_m2_t5_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19]],
        );
        let m2_t7_rstar = l3_m2_t7_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19], m2_a[20], m2_a[21], m2_a[22], m2_a[23], m2_a[24], m2_a[25], m2_a[26], m2_a[27]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19], m2_b[20], m2_b[21], m2_b[22], m2_b[23], m2_b[24], m2_b[25], m2_b[26], m2_b[27]],
        );
        let m2_t10_rstar = l3_m2_t10_rstar.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19], m2_a[20], m2_a[21], m2_a[22], m2_a[23], m2_a[24], m2_a[25], m2_a[26], m2_a[27], m2_a[28], m2_a[29], m2_a[30], m2_a[31], m2_a[32], m2_a[33], m2_a[34], m2_a[35], m2_a[36], m2_a[37], m2_a[38], m2_a[39]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19], m2_b[20], m2_b[21], m2_b[22], m2_b[23], m2_b[24], m2_b[25], m2_b[26], m2_b[27], m2_b[28], m2_b[29], m2_b[30], m2_b[31], m2_b[32], m2_b[33], m2_b[34], m2_b[35], m2_b[36], m2_b[37], m2_b[38], m2_b[39]],
        );

        // 2.2.4. Layer 4
        let m2_t1_pswaption = l4_m2_t1_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3]],
            m2_t1_rstar,
        );
        let m2_t2_pswaption = l4_m2_t2_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3], m2_vp[4], m2_vp[5], m2_vp[6], m2_vp[7]],
            m2_t2_rstar,
        );
        let m2_t3_pswaption = l4_m2_t3_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3], m2_vp[4], m2_vp[5], m2_vp[6], m2_vp[7], m2_vp[8], m2_vp[9], m2_vp[10], m2_vp[11]],
            m2_t3_rstar,
        );
        let m2_t5_pswaption = l4_m2_t5_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3], m2_vp[4], m2_vp[5], m2_vp[6], m2_vp[7], m2_vp[8], m2_vp[9], m2_vp[10], m2_vp[11], m2_vp[12], m2_vp[13], m2_vp[14], m2_vp[15], m2_vp[16], m2_vp[17], m2_vp[18], m2_vp[19]],
            m2_t5_rstar,
        );
        let m2_t7_pswaption = l4_m2_t7_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19], m2_a[20], m2_a[21], m2_a[22], m2_a[23], m2_a[24], m2_a[25], m2_a[26], m2_a[27]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19], m2_b[20], m2_b[21], m2_b[22], m2_b[23], m2_b[24], m2_b[25], m2_b[26], m2_b[27]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3], m2_vp[4], m2_vp[5], m2_vp[6], m2_vp[7], m2_vp[8], m2_vp[9], m2_vp[10], m2_vp[11], m2_vp[12], m2_vp[13], m2_vp[14], m2_vp[15], m2_vp[16], m2_vp[17], m2_vp[18], m2_vp[19], m2_vp[20], m2_vp[21], m2_vp[22], m2_vp[23], m2_vp[24], m2_vp[25], m2_vp[26], m2_vp[27]],
            m2_t7_rstar,
        );
        let m2_t10_pswaption = l4_m2_t10_pswaption.forward(
            [m2_a[0], m2_a[1], m2_a[2], m2_a[3], m2_a[4], m2_a[5], m2_a[6], m2_a[7], m2_a[8], m2_a[9], m2_a[10], m2_a[11], m2_a[12], m2_a[13], m2_a[14], m2_a[15], m2_a[16], m2_a[17], m2_a[18], m2_a[19], m2_a[20], m2_a[21], m2_a[22], m2_a[23], m2_a[24], m2_a[25], m2_a[26], m2_a[27], m2_a[28], m2_a[29], m2_a[30], m2_a[31], m2_a[32], m2_a[33], m2_a[34], m2_a[35], m2_a[36], m2_a[37], m2_a[38], m2_a[39]],
            [m2_b[0], m2_b[1], m2_b[2], m2_b[3], m2_b[4], m2_b[5], m2_b[6], m2_b[7], m2_b[8], m2_b[9], m2_b[10], m2_b[11], m2_b[12], m2_b[13], m2_b[14], m2_b[15], m2_b[16], m2_b[17], m2_b[18], m2_b[19], m2_b[20], m2_b[21], m2_b[22], m2_b[23], m2_b[24], m2_b[25], m2_b[26], m2_b[27], m2_b[28], m2_b[29], m2_b[30], m2_b[31], m2_b[32], m2_b[33], m2_b[34], m2_b[35], m2_b[36], m2_b[37], m2_b[38], m2_b[39]],
            [m2_vp[0], m2_vp[1], m2_vp[2], m2_vp[3], m2_vp[4], m2_vp[5], m2_vp[6], m2_vp[7], m2_vp[8], m2_vp[9], m2_vp[10], m2_vp[11], m2_vp[12], m2_vp[13], m2_vp[14], m2_vp[15], m2_vp[16], m2_vp[17], m2_vp[18], m2_vp[19], m2_vp[20], m2_vp[21], m2_vp[22], m2_vp[23], m2_vp[24], m2_vp[25], m2_vp[26], m2_vp[27], m2_vp[28], m2_vp[29], m2_vp[30], m2_vp[31], m2_vp[32], m2_vp[33], m2_vp[34], m2_vp[35], m2_vp[36], m2_vp[37], m2_vp[38], m2_vp[39]],
            m2_t10_rstar,
        );

        // 2.3. Option Maturity 3
        // 2.3.1. Layer 1
        let mut m3_b = [0.0; 40];
        for i in 0..40 {
            m3_b[i] = l1_m3_b[i].forward(alpha10, alpha20);
        }
        let m3_vr = l1_m3_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);

        // 2.3.2. Layer 2
        let mut m3_a = [0.0; 40];
        let mut m3_vp = [0.0; 40];
        for i in 0..40 {
            m3_a[i] = l2_m3_a[i].forward(m3_b[i], m3_vr);
            m3_vp[i] = l2_m3_vp[i].forward(m3_b[i], m3_vr);
        }

        // 2.3.3. Layer 3
        let m3_t1_rstar = l3_m3_t1_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3]],
        );
        let m3_t2_rstar = l3_m3_t2_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7]],
        );
        let m3_t3_rstar = l3_m3_t3_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11]],
        );
        let m3_t5_rstar = l3_m3_t5_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19]],
        );
        let m3_t7_rstar = l3_m3_t7_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19], m3_a[20], m3_a[21], m3_a[22], m3_a[23], m3_a[24], m3_a[25], m3_a[26], m3_a[27]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19], m3_b[20], m3_b[21], m3_b[22], m3_b[23], m3_b[24], m3_b[25], m3_b[26], m3_b[27]],
        );
        let m3_t10_rstar = l3_m3_t10_rstar.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19], m3_a[20], m3_a[21], m3_a[22], m3_a[23], m3_a[24], m3_a[25], m3_a[26], m3_a[27], m3_a[28], m3_a[29], m3_a[30], m3_a[31], m3_a[32], m3_a[33], m3_a[34], m3_a[35], m3_a[36], m3_a[37], m3_a[38], m3_a[39]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19], m3_b[20], m3_b[21], m3_b[22], m3_b[23], m3_b[24], m3_b[25], m3_b[26], m3_b[27], m3_b[28], m3_b[29], m3_b[30], m3_b[31], m3_b[32], m3_b[33], m3_b[34], m3_b[35], m3_b[36], m3_b[37], m3_b[38], m3_b[39]],
        );

        // 2.3.4. Layer 4
        let m3_t1_pswaption = l4_m3_t1_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3]],
            m3_t1_rstar,
        );
        let m3_t2_pswaption = l4_m3_t2_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3], m3_vp[4], m3_vp[5], m3_vp[6], m3_vp[7]],
            m3_t2_rstar,
        );
        let m3_t3_pswaption = l4_m3_t3_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3], m3_vp[4], m3_vp[5], m3_vp[6], m3_vp[7], m3_vp[8], m3_vp[9], m3_vp[10], m3_vp[11]],
            m3_t3_rstar,
        );
        let m3_t5_pswaption = l4_m3_t5_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3], m3_vp[4], m3_vp[5], m3_vp[6], m3_vp[7], m3_vp[8], m3_vp[9], m3_vp[10], m3_vp[11], m3_vp[12], m3_vp[13], m3_vp[14], m3_vp[15], m3_vp[16], m3_vp[17], m3_vp[18], m3_vp[19]],
            m3_t5_rstar,
        );
        let m3_t7_pswaption = l4_m3_t7_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19], m3_a[20], m3_a[21], m3_a[22], m3_a[23], m3_a[24], m3_a[25], m3_a[26], m3_a[27]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19], m3_b[20], m3_b[21], m3_b[22], m3_b[23], m3_b[24], m3_b[25], m3_b[26], m3_b[27]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3], m3_vp[4], m3_vp[5], m3_vp[6], m3_vp[7], m3_vp[8], m3_vp[9], m3_vp[10], m3_vp[11], m3_vp[12], m3_vp[13], m3_vp[14], m3_vp[15], m3_vp[16], m3_vp[17], m3_vp[18], m3_vp[19], m3_vp[20], m3_vp[21], m3_vp[22], m3_vp[23], m3_vp[24], m3_vp[25], m3_vp[26], m3_vp[27]],
            m3_t7_rstar,
        );
        let m3_t10_pswaption = l4_m3_t10_pswaption.forward(
            [m3_a[0], m3_a[1], m3_a[2], m3_a[3], m3_a[4], m3_a[5], m3_a[6], m3_a[7], m3_a[8], m3_a[9], m3_a[10], m3_a[11], m3_a[12], m3_a[13], m3_a[14], m3_a[15], m3_a[16], m3_a[17], m3_a[18], m3_a[19], m3_a[20], m3_a[21], m3_a[22], m3_a[23], m3_a[24], m3_a[25], m3_a[26], m3_a[27], m3_a[28], m3_a[29], m3_a[30], m3_a[31], m3_a[32], m3_a[33], m3_a[34], m3_a[35], m3_a[36], m3_a[37], m3_a[38], m3_a[39]],
            [m3_b[0], m3_b[1], m3_b[2], m3_b[3], m3_b[4], m3_b[5], m3_b[6], m3_b[7], m3_b[8], m3_b[9], m3_b[10], m3_b[11], m3_b[12], m3_b[13], m3_b[14], m3_b[15], m3_b[16], m3_b[17], m3_b[18], m3_b[19], m3_b[20], m3_b[21], m3_b[22], m3_b[23], m3_b[24], m3_b[25], m3_b[26], m3_b[27], m3_b[28], m3_b[29], m3_b[30], m3_b[31], m3_b[32], m3_b[33], m3_b[34], m3_b[35], m3_b[36], m3_b[37], m3_b[38], m3_b[39]],
            [m3_vp[0], m3_vp[1], m3_vp[2], m3_vp[3], m3_vp[4], m3_vp[5], m3_vp[6], m3_vp[7], m3_vp[8], m3_vp[9], m3_vp[10], m3_vp[11], m3_vp[12], m3_vp[13], m3_vp[14], m3_vp[15], m3_vp[16], m3_vp[17], m3_vp[18], m3_vp[19], m3_vp[20], m3_vp[21], m3_vp[22], m3_vp[23], m3_vp[24], m3_vp[25], m3_vp[26], m3_vp[27], m3_vp[28], m3_vp[29], m3_vp[30], m3_vp[31], m3_vp[32], m3_vp[33], m3_vp[34], m3_vp[35], m3_vp[36], m3_vp[37], m3_vp[38], m3_vp[39]],
            m3_t10_rstar,
        );

        // 2.4. Option Maturity 5
        // 2.4.1. Layer 1
        let mut m5_b = [0.0; 40];
        for i in 0..40 {
            m5_b[i] = l1_m5_b[i].forward(alpha10, alpha20);
        }
        let m5_vr = l1_m5_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);

        // 2.4.2. Layer 2
        let mut m5_a = [0.0; 40];
        let mut m5_vp = [0.0; 40];
        for i in 0..40 {
            m5_a[i] = l2_m5_a[i].forward(m5_b[i], m5_vr);
            m5_vp[i] = l2_m5_vp[i].forward(m5_b[i], m5_vr);
        }

        // 2.4.3. Layer 3
        let m5_t1_rstar = l3_m5_t1_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3]],
        );
        let m5_t2_rstar = l3_m5_t2_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7]],
        );
        let m5_t3_rstar = l3_m5_t3_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11]],
        );
        let m5_t5_rstar = l3_m5_t5_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19]],
        );
        let m5_t7_rstar = l3_m5_t7_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19], m5_a[20], m5_a[21], m5_a[22], m5_a[23], m5_a[24], m5_a[25], m5_a[26], m5_a[27]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19], m5_b[20], m5_b[21], m5_b[22], m5_b[23], m5_b[24], m5_b[25], m5_b[26], m5_b[27]],
        );
        let m5_t10_rstar = l3_m5_t10_rstar.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19], m5_a[20], m5_a[21], m5_a[22], m5_a[23], m5_a[24], m5_a[25], m5_a[26], m5_a[27], m5_a[28], m5_a[29], m5_a[30], m5_a[31], m5_a[32], m5_a[33], m5_a[34], m5_a[35], m5_a[36], m5_a[37], m5_a[38], m5_a[39]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19], m5_b[20], m5_b[21], m5_b[22], m5_b[23], m5_b[24], m5_b[25], m5_b[26], m5_b[27], m5_b[28], m5_b[29], m5_b[30], m5_b[31], m5_b[32], m5_b[33], m5_b[34], m5_b[35], m5_b[36], m5_b[37], m5_b[38], m5_b[39]],
        );

        // 2.4.4. Layer 4
        let m5_t1_pswaption = l4_m5_t1_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3]],
            m5_t1_rstar,
        );
        let m5_t2_pswaption = l4_m5_t2_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3], m5_vp[4], m5_vp[5], m5_vp[6], m5_vp[7]],
            m5_t2_rstar,
        );
        let m5_t3_pswaption = l4_m5_t3_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3], m5_vp[4], m5_vp[5], m5_vp[6], m5_vp[7], m5_vp[8], m5_vp[9], m5_vp[10], m5_vp[11]],
            m5_t3_rstar,
        );
        let m5_t5_pswaption = l4_m5_t5_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3], m5_vp[4], m5_vp[5], m5_vp[6], m5_vp[7], m5_vp[8], m5_vp[9], m5_vp[10], m5_vp[11], m5_vp[12], m5_vp[13], m5_vp[14], m5_vp[15], m5_vp[16], m5_vp[17], m5_vp[18], m5_vp[19]],
            m5_t5_rstar,
        );
        let m5_t7_pswaption = l4_m5_t7_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19], m5_a[20], m5_a[21], m5_a[22], m5_a[23], m5_a[24], m5_a[25], m5_a[26], m5_a[27]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19], m5_b[20], m5_b[21], m5_b[22], m5_b[23], m5_b[24], m5_b[25], m5_b[26], m5_b[27]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3], m5_vp[4], m5_vp[5], m5_vp[6], m5_vp[7], m5_vp[8], m5_vp[9], m5_vp[10], m5_vp[11], m5_vp[12], m5_vp[13], m5_vp[14], m5_vp[15], m5_vp[16], m5_vp[17], m5_vp[18], m5_vp[19], m5_vp[20], m5_vp[21], m5_vp[22], m5_vp[23], m5_vp[24], m5_vp[25], m5_vp[26], m5_vp[27]],
            m5_t7_rstar,
        );
        let m5_t10_pswaption = l4_m5_t10_pswaption.forward(
            [m5_a[0], m5_a[1], m5_a[2], m5_a[3], m5_a[4], m5_a[5], m5_a[6], m5_a[7], m5_a[8], m5_a[9], m5_a[10], m5_a[11], m5_a[12], m5_a[13], m5_a[14], m5_a[15], m5_a[16], m5_a[17], m5_a[18], m5_a[19], m5_a[20], m5_a[21], m5_a[22], m5_a[23], m5_a[24], m5_a[25], m5_a[26], m5_a[27], m5_a[28], m5_a[29], m5_a[30], m5_a[31], m5_a[32], m5_a[33], m5_a[34], m5_a[35], m5_a[36], m5_a[37], m5_a[38], m5_a[39]],
            [m5_b[0], m5_b[1], m5_b[2], m5_b[3], m5_b[4], m5_b[5], m5_b[6], m5_b[7], m5_b[8], m5_b[9], m5_b[10], m5_b[11], m5_b[12], m5_b[13], m5_b[14], m5_b[15], m5_b[16], m5_b[17], m5_b[18], m5_b[19], m5_b[20], m5_b[21], m5_b[22], m5_b[23], m5_b[24], m5_b[25], m5_b[26], m5_b[27], m5_b[28], m5_b[29], m5_b[30], m5_b[31], m5_b[32], m5_b[33], m5_b[34], m5_b[35], m5_b[36], m5_b[37], m5_b[38], m5_b[39]],
            [m5_vp[0], m5_vp[1], m5_vp[2], m5_vp[3], m5_vp[4], m5_vp[5], m5_vp[6], m5_vp[7], m5_vp[8], m5_vp[9], m5_vp[10], m5_vp[11], m5_vp[12], m5_vp[13], m5_vp[14], m5_vp[15], m5_vp[16], m5_vp[17], m5_vp[18], m5_vp[19], m5_vp[20], m5_vp[21], m5_vp[22], m5_vp[23], m5_vp[24], m5_vp[25], m5_vp[26], m5_vp[27], m5_vp[28], m5_vp[29], m5_vp[30], m5_vp[31], m5_vp[32], m5_vp[33], m5_vp[34], m5_vp[35], m5_vp[36], m5_vp[37], m5_vp[38], m5_vp[39]],
            m5_t10_rstar,
        );

        // 2.5. Option Maturity 7
        // 2.5.1. Layer 1
        let mut m7_b = [0.0; 40];
        for i in 0..40 {
            m7_b[i] = l1_m7_b[i].forward(alpha10, alpha20);
        }
        let m7_vr = l1_m7_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);

        // 2.5.2. Layer 2
        let mut m7_a = [0.0; 40];
        let mut m7_vp = [0.0; 40];
        for i in 0..40 {
            m7_a[i] = l2_m7_a[i].forward(m7_b[i], m7_vr);
            m7_vp[i] = l2_m7_vp[i].forward(m7_b[i], m7_vr);
        }

        // 2.5.3. Layer 3
        let m7_t1_rstar = l3_m7_t1_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3]],
        );
        let m7_t2_rstar = l3_m7_t2_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7]],
        );
        let m7_t3_rstar = l3_m7_t3_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11]],
        );
        let m7_t5_rstar = l3_m7_t5_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19]],
        );
        let m7_t7_rstar = l3_m7_t7_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19], m7_a[20], m7_a[21], m7_a[22], m7_a[23], m7_a[24], m7_a[25], m7_a[26], m7_a[27]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19], m7_b[20], m7_b[21], m7_b[22], m7_b[23], m7_b[24], m7_b[25], m7_b[26], m7_b[27]],
        );
        let m7_t10_rstar = l3_m7_t10_rstar.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19], m7_a[20], m7_a[21], m7_a[22], m7_a[23], m7_a[24], m7_a[25], m7_a[26], m7_a[27], m7_a[28], m7_a[29], m7_a[30], m7_a[31], m7_a[32], m7_a[33], m7_a[34], m7_a[35], m7_a[36], m7_a[37], m7_a[38], m7_a[39]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19], m7_b[20], m7_b[21], m7_b[22], m7_b[23], m7_b[24], m7_b[25], m7_b[26], m7_b[27], m7_b[28], m7_b[29], m7_b[30], m7_b[31], m7_b[32], m7_b[33], m7_b[34], m7_b[35], m7_b[36], m7_b[37], m7_b[38], m7_b[39]],
        );

        // 2.5.4. Layer 4
        let m7_t1_pswaption = l4_m7_t1_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3]],
            m7_t1_rstar,
        );
        let m7_t2_pswaption = l4_m7_t2_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3], m7_vp[4], m7_vp[5], m7_vp[6], m7_vp[7]],
            m7_t2_rstar,
        );
        let m7_t3_pswaption = l4_m7_t3_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3], m7_vp[4], m7_vp[5], m7_vp[6], m7_vp[7], m7_vp[8], m7_vp[9], m7_vp[10], m7_vp[11]],
            m7_t3_rstar,
        );
        let m7_t5_pswaption = l4_m7_t5_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3], m7_vp[4], m7_vp[5], m7_vp[6], m7_vp[7], m7_vp[8], m7_vp[9], m7_vp[10], m7_vp[11], m7_vp[12], m7_vp[13], m7_vp[14], m7_vp[15], m7_vp[16], m7_vp[17], m7_vp[18], m7_vp[19]],
            m7_t5_rstar,
        );
        let m7_t7_pswaption = l4_m7_t7_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19], m7_a[20], m7_a[21], m7_a[22], m7_a[23], m7_a[24], m7_a[25], m7_a[26], m7_a[27]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19], m7_b[20], m7_b[21], m7_b[22], m7_b[23], m7_b[24], m7_b[25], m7_b[26], m7_b[27]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3], m7_vp[4], m7_vp[5], m7_vp[6], m7_vp[7], m7_vp[8], m7_vp[9], m7_vp[10], m7_vp[11], m7_vp[12], m7_vp[13], m7_vp[14], m7_vp[15], m7_vp[16], m7_vp[17], m7_vp[18], m7_vp[19], m7_vp[20], m7_vp[21], m7_vp[22], m7_vp[23], m7_vp[24], m7_vp[25], m7_vp[26], m7_vp[27]],
            m7_t7_rstar,
        );
        let m7_t10_pswaption = l4_m7_t10_pswaption.forward(
            [m7_a[0], m7_a[1], m7_a[2], m7_a[3], m7_a[4], m7_a[5], m7_a[6], m7_a[7], m7_a[8], m7_a[9], m7_a[10], m7_a[11], m7_a[12], m7_a[13], m7_a[14], m7_a[15], m7_a[16], m7_a[17], m7_a[18], m7_a[19], m7_a[20], m7_a[21], m7_a[22], m7_a[23], m7_a[24], m7_a[25], m7_a[26], m7_a[27], m7_a[28], m7_a[29], m7_a[30], m7_a[31], m7_a[32], m7_a[33], m7_a[34], m7_a[35], m7_a[36], m7_a[37], m7_a[38], m7_a[39]],
            [m7_b[0], m7_b[1], m7_b[2], m7_b[3], m7_b[4], m7_b[5], m7_b[6], m7_b[7], m7_b[8], m7_b[9], m7_b[10], m7_b[11], m7_b[12], m7_b[13], m7_b[14], m7_b[15], m7_b[16], m7_b[17], m7_b[18], m7_b[19], m7_b[20], m7_b[21], m7_b[22], m7_b[23], m7_b[24], m7_b[25], m7_b[26], m7_b[27], m7_b[28], m7_b[29], m7_b[30], m7_b[31], m7_b[32], m7_b[33], m7_b[34], m7_b[35], m7_b[36], m7_b[37], m7_b[38], m7_b[39]],
            [m7_vp[0], m7_vp[1], m7_vp[2], m7_vp[3], m7_vp[4], m7_vp[5], m7_vp[6], m7_vp[7], m7_vp[8], m7_vp[9], m7_vp[10], m7_vp[11], m7_vp[12], m7_vp[13], m7_vp[14], m7_vp[15], m7_vp[16], m7_vp[17], m7_vp[18], m7_vp[19], m7_vp[20], m7_vp[21], m7_vp[22], m7_vp[23], m7_vp[24], m7_vp[25], m7_vp[26], m7_vp[27], m7_vp[28], m7_vp[29], m7_vp[30], m7_vp[31], m7_vp[32], m7_vp[33], m7_vp[34], m7_vp[35], m7_vp[36], m7_vp[37], m7_vp[38], m7_vp[39]],
            m7_t10_rstar,
        );

        // 2.6. Option Maturity 10
        // 2.6.1. Layer 1
        let mut m10_b = [0.0; 40];
        for i in 0..40 {
            m10_b[i] = l1_m10_b[i].forward(alpha10, alpha20);
        }
        let m10_vr = l1_m10_vr.forward(alpha10, sigma1, sigma2, sigma3, sigma5, sigma7, sigma10);

        // 2.6.2. Layer 2
        let mut m10_a = [0.0; 40];
        let mut m10_vp = [0.0; 40];
        for i in 0..40 {
            m10_a[i] = l2_m10_a[i].forward(m10_b[i], m10_vr);
            m10_vp[i] = l2_m10_vp[i].forward(m10_b[i], m10_vr);
        }

        // 2.6.3. Layer 3
        let m10_t1_rstar = l3_m10_t1_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3]],
        );
        let m10_t2_rstar = l3_m10_t2_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7]],
        );
        let m10_t3_rstar = l3_m10_t3_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11]],
        );
        let m10_t5_rstar = l3_m10_t5_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19]],
        );
        let m10_t7_rstar = l3_m10_t7_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19], m10_a[20], m10_a[21], m10_a[22], m10_a[23], m10_a[24], m10_a[25], m10_a[26], m10_a[27]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19], m10_b[20], m10_b[21], m10_b[22], m10_b[23], m10_b[24], m10_b[25], m10_b[26], m10_b[27]],
        );
        let m10_t10_rstar = l3_m10_t10_rstar.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19], m10_a[20], m10_a[21], m10_a[22], m10_a[23], m10_a[24], m10_a[25], m10_a[26], m10_a[27], m10_a[28], m10_a[29], m10_a[30], m10_a[31], m10_a[32], m10_a[33], m10_a[34], m10_a[35], m10_a[36], m10_a[37], m10_a[38], m10_a[39]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19], m10_b[20], m10_b[21], m10_b[22], m10_b[23], m10_b[24], m10_b[25], m10_b[26], m10_b[27], m10_b[28], m10_b[29], m10_b[30], m10_b[31], m10_b[32], m10_b[33], m10_b[34], m10_b[35], m10_b[36], m10_b[37], m10_b[38], m10_b[39]],
        );

        // 2.6.4. Layer 4
        let m10_t1_pswaption = l4_m10_t1_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3]],
            m10_t1_rstar,
        );
        let m10_t2_pswaption = l4_m10_t2_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3], m10_vp[4], m10_vp[5], m10_vp[6], m10_vp[7]],
            m10_t2_rstar,
        );
        let m10_t3_pswaption = l4_m10_t3_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3], m10_vp[4], m10_vp[5], m10_vp[6], m10_vp[7], m10_vp[8], m10_vp[9], m10_vp[10], m10_vp[11]],
            m10_t3_rstar,
        );
        let m10_t5_pswaption = l4_m10_t5_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3], m10_vp[4], m10_vp[5], m10_vp[6], m10_vp[7], m10_vp[8], m10_vp[9], m10_vp[10], m10_vp[11], m10_vp[12], m10_vp[13], m10_vp[14], m10_vp[15], m10_vp[16], m10_vp[17], m10_vp[18], m10_vp[19]],
            m10_t5_rstar,
        );
        let m10_t7_pswaption = l4_m10_t7_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19], m10_a[20], m10_a[21], m10_a[22], m10_a[23], m10_a[24], m10_a[25], m10_a[26], m10_a[27]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19], m10_b[20], m10_b[21], m10_b[22], m10_b[23], m10_b[24], m10_b[25], m10_b[26], m10_b[27]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3], m10_vp[4], m10_vp[5], m10_vp[6], m10_vp[7], m10_vp[8], m10_vp[9], m10_vp[10], m10_vp[11], m10_vp[12], m10_vp[13], m10_vp[14], m10_vp[15], m10_vp[16], m10_vp[17], m10_vp[18], m10_vp[19], m10_vp[20], m10_vp[21], m10_vp[22], m10_vp[23], m10_vp[24], m10_vp[25], m10_vp[26], m10_vp[27]],
            m10_t7_rstar,
        );
        let m10_t10_pswaption = l4_m10_t10_pswaption.forward(
            [m10_a[0], m10_a[1], m10_a[2], m10_a[3], m10_a[4], m10_a[5], m10_a[6], m10_a[7], m10_a[8], m10_a[9], m10_a[10], m10_a[11], m10_a[12], m10_a[13], m10_a[14], m10_a[15], m10_a[16], m10_a[17], m10_a[18], m10_a[19], m10_a[20], m10_a[21], m10_a[22], m10_a[23], m10_a[24], m10_a[25], m10_a[26], m10_a[27], m10_a[28], m10_a[29], m10_a[30], m10_a[31], m10_a[32], m10_a[33], m10_a[34], m10_a[35], m10_a[36], m10_a[37], m10_a[38], m10_a[39]],
            [m10_b[0], m10_b[1], m10_b[2], m10_b[3], m10_b[4], m10_b[5], m10_b[6], m10_b[7], m10_b[8], m10_b[9], m10_b[10], m10_b[11], m10_b[12], m10_b[13], m10_b[14], m10_b[15], m10_b[16], m10_b[17], m10_b[18], m10_b[19], m10_b[20], m10_b[21], m10_b[22], m10_b[23], m10_b[24], m10_b[25], m10_b[26], m10_b[27], m10_b[28], m10_b[29], m10_b[30], m10_b[31], m10_b[32], m10_b[33], m10_b[34], m10_b[35], m10_b[36], m10_b[37], m10_b[38], m10_b[39]],
            [m10_vp[0], m10_vp[1], m10_vp[2], m10_vp[3], m10_vp[4], m10_vp[5], m10_vp[6], m10_vp[7], m10_vp[8], m10_vp[9], m10_vp[10], m10_vp[11], m10_vp[12], m10_vp[13], m10_vp[14], m10_vp[15], m10_vp[16], m10_vp[17], m10_vp[18], m10_vp[19], m10_vp[20], m10_vp[21], m10_vp[22], m10_vp[23], m10_vp[24], m10_vp[25], m10_vp[26], m10_vp[27], m10_vp[28], m10_vp[29], m10_vp[30], m10_vp[31], m10_vp[32], m10_vp[33], m10_vp[34], m10_vp[35], m10_vp[36], m10_vp[37], m10_vp[38], m10_vp[39]],
            m10_t10_rstar,
        );
        

        // 2.7. Layer 5
        let pswaption_hw = [[m1_t1_pswaption, m1_t2_pswaption, m1_t3_pswaption, m1_t5_pswaption, m1_t7_pswaption, m1_t10_pswaption],
        [m2_t1_pswaption, m2_t2_pswaption, m2_t3_pswaption, m2_t5_pswaption, m2_t7_pswaption, m2_t10_pswaption],
        [m3_t1_pswaption, m3_t2_pswaption, m3_t3_pswaption, m3_t5_pswaption, m3_t7_pswaption, m3_t10_pswaption],
        [m5_t1_pswaption, m5_t2_pswaption, m5_t3_pswaption, m5_t5_pswaption, m5_t7_pswaption, m5_t10_pswaption],
        [m7_t1_pswaption, m7_t2_pswaption, m7_t3_pswaption, m7_t5_pswaption, m7_t7_pswaption, m7_t10_pswaption],
        [m10_t1_pswaption, m10_t2_pswaption, m10_t3_pswaption, m10_t5_pswaption, m10_t7_pswaption, m10_t10_pswaption]];

        let mrse = l5_mrse.forward(pswaption_hw);
        // let mrae = l5_mrae.forward(pswaption_hw);
        
        // 3. Backpropagation
        let mut dalpha10 = 0.0;
        let mut dalpha20 = 0.0;
        let mut dsigma1 = 0.0;
        let mut dsigma2 = 0.0;
        let mut dsigma3 = 0.0;
        let mut dsigma5 = 0.0;
        let mut dsigma7 = 0.0;
        let mut dsigma10 = 0.0;
        
        // 3.1. Layer 5 (OK)
        let l4_dpswaption = l5_mrse.backward(1.0);
        // let l4_dpswaption = l5_mrae.backward(1.0);
        
        // 3.2. Option Maturity 1
        // 3.2.1. Layer 4
        let (l3_m1_t1_da, l3_m1_t1_db, l3_m1_t1_dvp, l3_m1_t1_drstar) = l4_m1_t1_pswaption.backward(l4_dpswaption[0][0]);
        let (l3_m1_t2_da, l3_m1_t2_db, l3_m1_t2_dvp, l3_m1_t2_drstar) = l4_m1_t2_pswaption.backward(l4_dpswaption[0][1]);
        let (l3_m1_t3_da, l3_m1_t3_db, l3_m1_t3_dvp, l3_m1_t3_drstar) = l4_m1_t3_pswaption.backward(l4_dpswaption[0][2]);
        let (l3_m1_t5_da, l3_m1_t5_db, l3_m1_t5_dvp, l3_m1_t5_drstar) = l4_m1_t5_pswaption.backward(l4_dpswaption[0][3]);
        let (l3_m1_t7_da, l3_m1_t7_db, l3_m1_t7_dvp, l3_m1_t7_drstar) = l4_m1_t7_pswaption.backward(l4_dpswaption[0][4]);
        let (l3_m1_t10_da, l3_m1_t10_db, l3_m1_t10_dvp, l3_m1_t10_drstar) = l4_m1_t10_pswaption.backward(l4_dpswaption[0][5]);
        
        
        // 3.2.2. Layer 3 (OK)
        let (l2_m1_t1_da, l2_m1_t1_db) = l3_m1_t1_rstar.backward(l3_m1_t1_drstar);
        let (l2_m1_t2_da, l2_m1_t2_db) = l3_m1_t2_rstar.backward(l3_m1_t2_drstar);
        let (l2_m1_t3_da, l2_m1_t3_db) = l3_m1_t3_rstar.backward(l3_m1_t3_drstar);
        let (l2_m1_t5_da, l2_m1_t5_db) = l3_m1_t5_rstar.backward(l3_m1_t5_drstar);
        let (l2_m1_t7_da, l2_m1_t7_db) = l3_m1_t7_rstar.backward(l3_m1_t7_drstar);
        let (l2_m1_t10_da, l2_m1_t10_db) = l3_m1_t10_rstar.backward(l3_m1_t10_drstar);
        
        
        // 3.2.3. Layer 2
        let mut l2_m1_da = [0.0; 40];
        let mut l2_m1_dvp = [0.0; 40];
        let mut l1_m1_db = [0.0; 40];
        let mut l1_m1_dvr = 0.0;
        
        for i in 0..4 { l2_m1_da[i] += l2_m1_t1_da[i] + l3_m1_t1_da[i]; }
        for i in 0..8 { l2_m1_da[i] += l2_m1_t2_da[i] + l3_m1_t2_da[i]; }
        for i in 0..12 { l2_m1_da[i] += l2_m1_t3_da[i] + l3_m1_t3_da[i]; }
        for i in 0..20 { l2_m1_da[i] += l2_m1_t5_da[i] + l3_m1_t5_da[i]; }
        for i in 0..28 { l2_m1_da[i] += l2_m1_t7_da[i] + l3_m1_t7_da[i]; }
        for i in 0..40 { l2_m1_da[i] += l2_m1_t10_da[i] + l3_m1_t10_da[i]; }
        
        for i in 0..4 { l2_m1_dvp[i] += l3_m1_t1_dvp[i]; }
        for i in 0..8 { l2_m1_dvp[i] += l3_m1_t2_dvp[i]; }
        for i in 0..12 { l2_m1_dvp[i] += l3_m1_t3_dvp[i]; }
        for i in 0..20 { l2_m1_dvp[i] += l3_m1_t5_dvp[i]; }
        for i in 0..28 { l2_m1_dvp[i] += l3_m1_t7_dvp[i]; }
        for i in 0..40 { l2_m1_dvp[i] += l3_m1_t10_dvp[i]; }
        
        for i in 0..40 {
            let tmp = l2_m1_a[i].backward(l2_m1_da[i]);
            l1_m1_db[i] += tmp.0;
            l1_m1_dvr += tmp.1;
            let tmp = l2_m1_vp[i].backward(l2_m1_dvp[i]);
            l1_m1_db[i] += tmp.0;
            l1_m1_dvr += tmp.1;
        }
        
        // 3.2.4. Layer 1
        for i in 0..4 { l1_m1_db[i] += l2_m1_t1_db[i] + l3_m1_t1_db[i]; }
        for i in 0..8 { l1_m1_db[i] += l2_m1_t2_db[i] + l3_m1_t2_db[i]; }
        for i in 0..12 { l1_m1_db[i] += l2_m1_t3_db[i] + l3_m1_t3_db[i]; }
        for i in 0..20 { l1_m1_db[i] += l2_m1_t5_db[i] + l3_m1_t5_db[i]; }
        for i in 0..28 { l1_m1_db[i] += l2_m1_t7_db[i] + l3_m1_t7_db[i]; }
        for i in 0..40 { l1_m1_db[i] += l2_m1_t10_db[i] + l3_m1_t10_db[i]; }
        
        for i in 0..40 {
            let tmp = l1_m1_b[i].backward(l1_m1_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m1_vr.backward(l1_m1_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        // 3.3. Option Maturity 2
        // 3.3.1. Layer 4
        let (l3_m2_t1_da, l3_m2_t1_db, l3_m2_t1_dvp, l3_m2_t1_drstar) = l4_m2_t1_pswaption.backward(l4_dpswaption[1][0]);
        let (l3_m2_t2_da, l3_m2_t2_db, l3_m2_t2_dvp, l3_m2_t2_drstar) = l4_m2_t2_pswaption.backward(l4_dpswaption[1][1]);
        let (l3_m2_t3_da, l3_m2_t3_db, l3_m2_t3_dvp, l3_m2_t3_drstar) = l4_m2_t3_pswaption.backward(l4_dpswaption[1][2]);
        let (l3_m2_t5_da, l3_m2_t5_db, l3_m2_t5_dvp, l3_m2_t5_drstar) = l4_m2_t5_pswaption.backward(l4_dpswaption[1][3]);
        let (l3_m2_t7_da, l3_m2_t7_db, l3_m2_t7_dvp, l3_m2_t7_drstar) = l4_m2_t7_pswaption.backward(l4_dpswaption[1][4]);
        let (l3_m2_t10_da, l3_m2_t10_db, l3_m2_t10_dvp, l3_m2_t10_drstar) = l4_m2_t10_pswaption.backward(l4_dpswaption[1][5]);
        
        // 3.3.2. Layer 3
        let (l2_m2_t1_da, l2_m2_t1_db) = l3_m2_t1_rstar.backward(l3_m2_t1_drstar);
        let (l2_m2_t2_da, l2_m2_t2_db) = l3_m2_t2_rstar.backward(l3_m2_t2_drstar);
        let (l2_m2_t3_da, l2_m2_t3_db) = l3_m2_t3_rstar.backward(l3_m2_t3_drstar);
        let (l2_m2_t5_da, l2_m2_t5_db) = l3_m2_t5_rstar.backward(l3_m2_t5_drstar);
        let (l2_m2_t7_da, l2_m2_t7_db) = l3_m2_t7_rstar.backward(l3_m2_t7_drstar);
        let (l2_m2_t10_da, l2_m2_t10_db) = l3_m2_t10_rstar.backward(l3_m2_t10_drstar);
        
        // 3.3.3. Layer 2
        let mut l2_m2_da = [0.0; 40];
        let mut l2_m2_dvp = [0.0; 40];
        let mut l1_m2_db = [0.0; 40];
        let mut l1_m2_dvr = 0.0;
        
        for i in 0..4 { l2_m2_da[i] += l2_m2_t1_da[i] + l3_m2_t1_da[i]; }
        for i in 0..8 { l2_m2_da[i] += l2_m2_t2_da[i] + l3_m2_t2_da[i]; }
        for i in 0..12 { l2_m2_da[i] += l2_m2_t3_da[i] + l3_m2_t3_da[i]; }
        for i in 0..20 { l2_m2_da[i] += l2_m2_t5_da[i] + l3_m2_t5_da[i]; }
        for i in 0..28 { l2_m2_da[i] += l2_m2_t7_da[i] + l3_m2_t7_da[i]; }
        for i in 0..40 { l2_m2_da[i] += l2_m2_t10_da[i] + l3_m2_t10_da[i]; }
        
        for i in 0..4 { l2_m2_dvp[i] += l3_m2_t1_dvp[i]; }
        for i in 0..8 { l2_m2_dvp[i] += l3_m2_t2_dvp[i]; }
        for i in 0..12 { l2_m2_dvp[i] += l3_m2_t3_dvp[i]; }
        for i in 0..20 { l2_m2_dvp[i] += l3_m2_t5_dvp[i]; }
        for i in 0..28 { l2_m2_dvp[i] += l3_m2_t7_dvp[i]; }
        for i in 0..40 { l2_m2_dvp[i] += l3_m2_t10_dvp[i]; }
        
        for i in 0..40 {
            let tmp = l2_m2_a[i].backward(l2_m2_da[i]);
            l1_m2_db[i] += tmp.0;
            l1_m2_dvr += tmp.1;
            let tmp = l2_m2_vp[i].backward(l2_m2_dvp[i]);
            l1_m2_db[i] += tmp.0;
            l1_m2_dvr += tmp.1;
        }
        
        // 3.3.4. Layer 1
        for i in 0..4 { l1_m2_db[i] += l2_m2_t1_db[i] + l3_m2_t1_db[i]; }
        for i in 0..8 { l1_m2_db[i] += l2_m2_t2_db[i] + l3_m2_t2_db[i]; }
        for i in 0..12 { l1_m2_db[i] += l2_m2_t3_db[i] + l3_m2_t3_db[i]; }
        for i in 0..20 { l1_m2_db[i] += l2_m2_t5_db[i] + l3_m2_t5_db[i]; }
        for i in 0..28 { l1_m2_db[i] += l2_m2_t7_db[i] + l3_m2_t7_db[i]; }
        for i in 0..40 { l1_m2_db[i] += l2_m2_t10_db[i] + l3_m2_t10_db[i]; }
        
        for i in 0..40 {
            let tmp = l1_m2_b[i].backward(l1_m2_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m2_vr.backward(l1_m2_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        // 3.4. Option Maturity 3
        // 3.4.1. Layer 4
        let (l3_m3_t1_da, l3_m3_t1_db, l3_m3_t1_dvp, l3_m3_t1_drstar) = l4_m3_t1_pswaption.backward(l4_dpswaption[2][0]);
        let (l3_m3_t2_da, l3_m3_t2_db, l3_m3_t2_dvp, l3_m3_t2_drstar) = l4_m3_t2_pswaption.backward(l4_dpswaption[2][1]);
        let (l3_m3_t3_da, l3_m3_t3_db, l3_m3_t3_dvp, l3_m3_t3_drstar) = l4_m3_t3_pswaption.backward(l4_dpswaption[2][2]);
        let (l3_m3_t5_da, l3_m3_t5_db, l3_m3_t5_dvp, l3_m3_t5_drstar) = l4_m3_t5_pswaption.backward(l4_dpswaption[2][3]);
        let (l3_m3_t7_da, l3_m3_t7_db, l3_m3_t7_dvp, l3_m3_t7_drstar) = l4_m3_t7_pswaption.backward(l4_dpswaption[2][4]);
        let (l3_m3_t10_da, l3_m3_t10_db, l3_m3_t10_dvp, l3_m3_t10_drstar) = l4_m3_t10_pswaption.backward(l4_dpswaption[2][5]);
        
        // 3.4.2. Layer 3
        let (l2_m3_t1_da, l2_m3_t1_db) = l3_m3_t1_rstar.backward(l3_m3_t1_drstar);
        let (l2_m3_t2_da, l2_m3_t2_db) = l3_m3_t2_rstar.backward(l3_m3_t2_drstar);
        let (l2_m3_t3_da, l2_m3_t3_db) = l3_m3_t3_rstar.backward(l3_m3_t3_drstar);
        let (l2_m3_t5_da, l2_m3_t5_db) = l3_m3_t5_rstar.backward(l3_m3_t5_drstar);
        let (l2_m3_t7_da, l2_m3_t7_db) = l3_m3_t7_rstar.backward(l3_m3_t7_drstar);
        let (l2_m3_t10_da, l2_m3_t10_db) = l3_m3_t10_rstar.backward(l3_m3_t10_drstar);
        
        // 3.4.3. Layer 2
        let mut l2_m3_da = [0.0; 40];
        let mut l2_m3_dvp = [0.0; 40];
        let mut l1_m3_db = [0.0; 40];
        let mut l1_m3_dvr = 0.0;
        
        for i in 0..4 { l2_m3_da[i] += l2_m3_t1_da[i] + l3_m3_t1_da[i]; }
        for i in 0..8 { l2_m3_da[i] += l2_m3_t2_da[i] + l3_m3_t2_da[i]; }
        for i in 0..12 { l2_m3_da[i] += l2_m3_t3_da[i] + l3_m3_t3_da[i]; }
        for i in 0..20 { l2_m3_da[i] += l2_m3_t5_da[i] + l3_m3_t5_da[i]; }
        for i in 0..28 { l2_m3_da[i] += l2_m3_t7_da[i] + l3_m3_t7_da[i]; }
        for i in 0..40 { l2_m3_da[i] += l2_m3_t10_da[i] + l3_m3_t10_da[i]; }
        
        for i in 0..4 { l2_m3_dvp[i] += l3_m3_t1_dvp[i]; }
        for i in 0..8 { l2_m3_dvp[i] += l3_m3_t2_dvp[i]; }
        for i in 0..12 { l2_m3_dvp[i] += l3_m3_t3_dvp[i]; }
        for i in 0..20 { l2_m3_dvp[i] += l3_m3_t5_dvp[i]; }
        for i in 0..28 { l2_m3_dvp[i] += l3_m3_t7_dvp[i]; }
        for i in 0..40 { l2_m3_dvp[i] += l3_m3_t10_dvp[i]; }

        for i in 0..40 {
            let tmp = l2_m3_a[i].backward(l2_m3_da[i]);
            l1_m3_db[i] += tmp.0;
            l1_m3_dvr += tmp.1;
            let tmp = l2_m3_vp[i].backward(l2_m3_dvp[i]);
            l1_m3_db[i] += tmp.0;
            l1_m3_dvr += tmp.1;
        }
        
        // 3.4.4. Layer 1
        for i in 0..4 { l1_m3_db[i] += l2_m3_t1_db[i] + l3_m3_t1_db[i]; }
        for i in 0..8 { l1_m3_db[i] += l2_m3_t2_db[i] + l3_m3_t2_db[i]; }
        for i in 0..12 { l1_m3_db[i] += l2_m3_t3_db[i] + l3_m3_t3_db[i]; }
        for i in 0..20 { l1_m3_db[i] += l2_m3_t5_db[i] + l3_m3_t5_db[i]; }
        for i in 0..28 { l1_m3_db[i] += l2_m3_t7_db[i] + l3_m3_t7_db[i]; }
        for i in 0..40 { l1_m3_db[i] += l2_m3_t10_db[i] + l3_m3_t10_db[i]; }
        
        for i in 0..40 {
            let tmp = l1_m3_b[i].backward(l1_m3_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m3_vr.backward(l1_m3_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        // 3.5. Option Maturity 5
        // 3.5.1. Layer 4
        let (l3_m5_t1_da, l3_m5_t1_db, l3_m5_t1_dvp, l3_m5_t1_drstar) = l4_m5_t1_pswaption.backward(l4_dpswaption[3][0]);
        let (l3_m5_t2_da, l3_m5_t2_db, l3_m5_t2_dvp, l3_m5_t2_drstar) = l4_m5_t2_pswaption.backward(l4_dpswaption[3][1]);
        let (l3_m5_t3_da, l3_m5_t3_db, l3_m5_t3_dvp, l3_m5_t3_drstar) = l4_m5_t3_pswaption.backward(l4_dpswaption[3][2]);
        let (l3_m5_t5_da, l3_m5_t5_db, l3_m5_t5_dvp, l3_m5_t5_drstar) = l4_m5_t5_pswaption.backward(l4_dpswaption[3][3]);
        let (l3_m5_t7_da, l3_m5_t7_db, l3_m5_t7_dvp, l3_m5_t7_drstar) = l4_m5_t7_pswaption.backward(l4_dpswaption[3][4]);
        let (l3_m5_t10_da, l3_m5_t10_db, l3_m5_t10_dvp, l3_m5_t10_drstar) = l4_m5_t10_pswaption.backward(l4_dpswaption[3][5]);
        
        // 3.5.2. Layer 3
        let (l2_m5_t1_da, l2_m5_t1_db) = l3_m5_t1_rstar.backward(l3_m5_t1_drstar);
        let (l2_m5_t2_da, l2_m5_t2_db) = l3_m5_t2_rstar.backward(l3_m5_t2_drstar);
        let (l2_m5_t3_da, l2_m5_t3_db) = l3_m5_t3_rstar.backward(l3_m5_t3_drstar);
        let (l2_m5_t5_da, l2_m5_t5_db) = l3_m5_t5_rstar.backward(l3_m5_t5_drstar);
        let (l2_m5_t7_da, l2_m5_t7_db) = l3_m5_t7_rstar.backward(l3_m5_t7_drstar);
        let (l2_m5_t10_da, l2_m5_t10_db) = l3_m5_t10_rstar.backward(l3_m5_t10_drstar);
        
        // 3.5.3. Layer 2
        let mut l2_m5_da = [0.0; 40];
        let mut l2_m5_dvp = [0.0; 40];
        let mut l1_m5_db = [0.0; 40];
        let mut l1_m5_dvr = 0.0;
        
        for i in 0..4 { l2_m5_da[i] += l2_m5_t1_da[i] + l3_m5_t1_da[i]; }
        for i in 0..8 { l2_m5_da[i] += l2_m5_t2_da[i] + l3_m5_t2_da[i]; }
        for i in 0..12 { l2_m5_da[i] += l2_m5_t3_da[i] + l3_m5_t3_da[i]; }
        for i in 0..20 { l2_m5_da[i] += l2_m5_t5_da[i] + l3_m5_t5_da[i]; }
        for i in 0..28 { l2_m5_da[i] += l2_m5_t7_da[i] + l3_m5_t7_da[i]; }
        for i in 0..40 { l2_m5_da[i] += l2_m5_t10_da[i] + l3_m5_t10_da[i]; }
        
        for i in 0..4 { l2_m5_dvp[i] += l3_m5_t1_dvp[i]; }
        for i in 0..8 { l2_m5_dvp[i] += l3_m5_t2_dvp[i]; }
        for i in 0..12 { l2_m5_dvp[i] += l3_m5_t3_dvp[i]; }
        for i in 0..20 { l2_m5_dvp[i] += l3_m5_t5_dvp[i]; }
        for i in 0..28 { l2_m5_dvp[i] += l3_m5_t7_dvp[i]; }
        for i in 0..40 { l2_m5_dvp[i] += l3_m5_t10_dvp[i]; }
        
        for i in 0..40 {
            let tmp = l2_m5_a[i].backward(l2_m5_da[i]);
            l1_m5_db[i] += tmp.0;
            l1_m5_dvr += tmp.1;
            let tmp = l2_m5_vp[i].backward(l2_m5_dvp[i]);
            l1_m5_db[i] += tmp.0;
            l1_m5_dvr += tmp.1;
        }
        
        // 3.5.4. Layer 1
        for i in 0..4 { l1_m5_db[i] += l2_m5_t1_db[i] + l3_m5_t1_db[i]; }
        for i in 0..8 { l1_m5_db[i] += l2_m5_t2_db[i] + l3_m5_t2_db[i]; }
        for i in 0..12 { l1_m5_db[i] += l2_m5_t3_db[i] + l3_m5_t3_db[i]; }
        for i in 0..20 { l1_m5_db[i] += l2_m5_t5_db[i] + l3_m5_t5_db[i]; }
        for i in 0..28 { l1_m5_db[i] += l2_m5_t7_db[i] + l3_m5_t7_db[i]; }
        for i in 0..40 { l1_m5_db[i] += l2_m5_t10_db[i] + l3_m5_t10_db[i]; }
        
        for i in 0..40 {
            let tmp = l1_m5_b[i].backward(l1_m5_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m5_vr.backward(l1_m5_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        // 3.6. Option Maturity 7
        // 3.6.1. Layer 4
        let (l3_m7_t1_da, l3_m7_t1_db, l3_m7_t1_dvp, l3_m7_t1_drstar) = l4_m7_t1_pswaption.backward(l4_dpswaption[4][0]);
        let (l3_m7_t2_da, l3_m7_t2_db, l3_m7_t2_dvp, l3_m7_t2_drstar) = l4_m7_t2_pswaption.backward(l4_dpswaption[4][1]);
        let (l3_m7_t3_da, l3_m7_t3_db, l3_m7_t3_dvp, l3_m7_t3_drstar) = l4_m7_t3_pswaption.backward(l4_dpswaption[4][2]);
        let (l3_m7_t5_da, l3_m7_t5_db, l3_m7_t5_dvp, l3_m7_t5_drstar) = l4_m7_t5_pswaption.backward(l4_dpswaption[4][3]);
        let (l3_m7_t7_da, l3_m7_t7_db, l3_m7_t7_dvp, l3_m7_t7_drstar) = l4_m7_t7_pswaption.backward(l4_dpswaption[4][4]);
        let (l3_m7_t10_da, l3_m7_t10_db, l3_m7_t10_dvp, l3_m7_t10_drstar) = l4_m7_t10_pswaption.backward(l4_dpswaption[4][5]);
        
        // 3.6.2. Layer 3
        let (l2_m7_t1_da, l2_m7_t1_db) = l3_m7_t1_rstar.backward(l3_m7_t1_drstar);
        let (l2_m7_t2_da, l2_m7_t2_db) = l3_m7_t2_rstar.backward(l3_m7_t2_drstar);
        let (l2_m7_t3_da, l2_m7_t3_db) = l3_m7_t3_rstar.backward(l3_m7_t3_drstar);
        let (l2_m7_t5_da, l2_m7_t5_db) = l3_m7_t5_rstar.backward(l3_m7_t5_drstar);
        let (l2_m7_t7_da, l2_m7_t7_db) = l3_m7_t7_rstar.backward(l3_m7_t7_drstar);
        let (l2_m7_t10_da, l2_m7_t10_db) = l3_m7_t10_rstar.backward(l3_m7_t10_drstar);
        
        // 3.6.3. Layer 2
        let mut l2_m7_da = [0.0; 40];
        let mut l2_m7_dvp = [0.0; 40];
        let mut l1_m7_db = [0.0; 40];
        let mut l1_m7_dvr = 0.0;
        
        for i in 0..4 { l2_m7_da[i] += l2_m7_t1_da[i] + l3_m7_t1_da[i]; }
        for i in 0..8 { l2_m7_da[i] += l2_m7_t2_da[i] + l3_m7_t2_da[i]; }
        for i in 0..12 { l2_m7_da[i] += l2_m7_t3_da[i] + l3_m7_t3_da[i]; }
        for i in 0..20 { l2_m7_da[i] += l2_m7_t5_da[i] + l3_m7_t5_da[i]; }
        for i in 0..28 { l2_m7_da[i] += l2_m7_t7_da[i] + l3_m7_t7_da[i]; }
        for i in 0..40 { l2_m7_da[i] += l2_m7_t10_da[i] + l3_m7_t10_da[i]; }
        
        for i in 0..4 { l2_m7_dvp[i] += l3_m7_t1_dvp[i]; }
        for i in 0..8 { l2_m7_dvp[i] += l3_m7_t2_dvp[i]; }
        for i in 0..12 { l2_m7_dvp[i] += l3_m7_t3_dvp[i]; }
        for i in 0..20 { l2_m7_dvp[i] += l3_m7_t5_dvp[i]; }
        for i in 0..28 { l2_m7_dvp[i] += l3_m7_t7_dvp[i]; }
        for i in 0..40 { l2_m7_dvp[i] += l3_m7_t10_dvp[i]; }
        
        for i in 0..40 {
            let tmp = l2_m7_a[i].backward(l2_m7_da[i]);
            l1_m7_db[i] += tmp.0;
            l1_m7_dvr += tmp.1;
            let tmp = l2_m7_vp[i].backward(l2_m7_dvp[i]);
            l1_m7_db[i] += tmp.0;
            l1_m7_dvr += tmp.1;
        }
        
        // 3.6.4. Layer 1
        for i in 0..4 { l1_m7_db[i] += l2_m7_t1_db[i] + l3_m7_t1_db[i]; }
        for i in 0..8 { l1_m7_db[i] += l2_m7_t2_db[i] + l3_m7_t2_db[i]; }
        for i in 0..12 { l1_m7_db[i] += l2_m7_t3_db[i] + l3_m7_t3_db[i]; }
        for i in 0..20 { l1_m7_db[i] += l2_m7_t5_db[i] + l3_m7_t5_db[i]; }
        for i in 0..28 { l1_m7_db[i] += l2_m7_t7_db[i] + l3_m7_t7_db[i]; }
        for i in 0..40 { l1_m7_db[i] += l2_m7_t10_db[i] + l3_m7_t10_db[i]; }
        
        for i in 0..40 {
            let tmp = l1_m7_b[i].backward(l1_m7_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m7_vr.backward(l1_m7_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        // 3.7. Option Maturity 10
        // 3.7.1. Layer 4
        let (l3_m10_t1_da, l3_m10_t1_db, l3_m10_t1_dvp, l3_m10_t1_drstar) = l4_m10_t1_pswaption.backward(l4_dpswaption[5][0]);
        let (l3_m10_t2_da, l3_m10_t2_db, l3_m10_t2_dvp, l3_m10_t2_drstar) = l4_m10_t2_pswaption.backward(l4_dpswaption[5][1]);
        let (l3_m10_t3_da, l3_m10_t3_db, l3_m10_t3_dvp, l3_m10_t3_drstar) = l4_m10_t3_pswaption.backward(l4_dpswaption[5][2]);
        let (l3_m10_t5_da, l3_m10_t5_db, l3_m10_t5_dvp, l3_m10_t5_drstar) = l4_m10_t5_pswaption.backward(l4_dpswaption[5][3]);
        let (l3_m10_t7_da, l3_m10_t7_db, l3_m10_t7_dvp, l3_m10_t7_drstar) = l4_m10_t7_pswaption.backward(l4_dpswaption[5][4]);
        let (l3_m10_t10_da, l3_m10_t10_db, l3_m10_t10_dvp, l3_m10_t10_drstar) = l4_m10_t10_pswaption.backward(l4_dpswaption[5][5]);
        
        // 3.7.2. Layer 3
        let (l2_m10_t1_da, l2_m10_t1_db) = l3_m10_t1_rstar.backward(l3_m10_t1_drstar);
        let (l2_m10_t2_da, l2_m10_t2_db) = l3_m10_t2_rstar.backward(l3_m10_t2_drstar);
        let (l2_m10_t3_da, l2_m10_t3_db) = l3_m10_t3_rstar.backward(l3_m10_t3_drstar);
        let (l2_m10_t5_da, l2_m10_t5_db) = l3_m10_t5_rstar.backward(l3_m10_t5_drstar);
        let (l2_m10_t7_da, l2_m10_t7_db) = l3_m10_t7_rstar.backward(l3_m10_t7_drstar);
        let (l2_m10_t10_da, l2_m10_t10_db) = l3_m10_t10_rstar.backward(l3_m10_t10_drstar);
        
        // 3.7.3. Layer 2
        let mut l2_m10_da = [0.0; 40];
        let mut l2_m10_dvp = [0.0; 40];
        let mut l1_m10_db = [0.0; 40];
        let mut l1_m10_dvr = 0.0;
        
        for i in 0..4 { l2_m10_da[i] += l2_m10_t1_da[i] + l3_m10_t1_da[i]; }
        for i in 0..8 { l2_m10_da[i] += l2_m10_t2_da[i] + l3_m10_t2_da[i]; }
        for i in 0..12 { l2_m10_da[i] += l2_m10_t3_da[i] + l3_m10_t3_da[i]; }
        for i in 0..20 { l2_m10_da[i] += l2_m10_t5_da[i] + l3_m10_t5_da[i]; }
        for i in 0..28 { l2_m10_da[i] += l2_m10_t7_da[i] + l3_m10_t7_da[i]; }
        for i in 0..40 { l2_m10_da[i] += l2_m10_t10_da[i] + l3_m10_t10_da[i]; }
        
        for i in 0..4 { l2_m10_dvp[i] += l3_m10_t1_dvp[i]; }
        for i in 0..8 { l2_m10_dvp[i] += l3_m10_t2_dvp[i]; }
        for i in 0..12 { l2_m10_dvp[i] += l3_m10_t3_dvp[i]; }
        for i in 0..20 { l2_m10_dvp[i] += l3_m10_t5_dvp[i]; }
        for i in 0..28 { l2_m10_dvp[i] += l3_m10_t7_dvp[i]; }
        for i in 0..40 { l2_m10_dvp[i] += l3_m10_t10_dvp[i]; }
        
        for i in 0..40 {
            let tmp = l2_m10_a[i].backward(l2_m10_da[i]);
            l1_m10_db[i] += tmp.0;
            l1_m10_dvr += tmp.1;
            let tmp = l2_m10_vp[i].backward(l2_m10_dvp[i]);
            l1_m10_db[i] += tmp.0;
            l1_m10_dvr += tmp.1;
        }
        
        // 3.7.4. Layer 1
        for i in 0..4 { l1_m10_db[i] += l2_m10_t1_db[i] + l3_m10_t1_db[i]; }
        for i in 0..8 { l1_m10_db[i] += l2_m10_t2_db[i] + l3_m10_t2_db[i]; }
        for i in 0..12 { l1_m10_db[i] += l2_m10_t3_db[i] + l3_m10_t3_db[i]; }
        for i in 0..20 { l1_m10_db[i] += l2_m10_t5_db[i] + l3_m10_t5_db[i]; }
        for i in 0..28 { l1_m10_db[i] += l2_m10_t7_db[i] + l3_m10_t7_db[i]; }
        for i in 0..40 { l1_m10_db[i] += l2_m10_t10_db[i] + l3_m10_t10_db[i]; }
        
        // 3.8. Gradient
        for i in 0..40 {
            let tmp = l1_m10_b[i].backward(l1_m10_db[i]);
            dalpha10 += tmp.0;
            dalpha20 += tmp.1;
        }
        let tmp = l1_m10_vr.backward(l1_m10_dvr);
        dalpha10 += tmp.0;
        dsigma1 += tmp.1;
        dsigma2 += tmp.2;
        dsigma3 += tmp.3;
        dsigma5 += tmp.4;
        dsigma7 += tmp.5;
        dsigma10 += tmp.6;
        
        (mrse, [dalpha10, dalpha20, dsigma1, dsigma2, dsigma3, dsigma5, dsigma7, dsigma10])
    };
    
    gd(step, p0, lr, tol)
}
