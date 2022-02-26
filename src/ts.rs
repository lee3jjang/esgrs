#[derive(Debug, Copy, Clone)]
pub struct TermStructure {
    pub p: [f64; 241],
    pub f: [f64; 241],
}

fn ytm_price(t: f64, ytm: f64, freq: f64) -> f64 {
    if freq == 0.0 {
        1.0/ f64::powf(1.0 + ytm, t)
    } else {
        let dt = 1.0/freq;
        let mut p = 0.0;
        let mut ti = t;
        let mut cf: f64;
        let mut df: f64;

        while ti > 0.0 {
            if ti == t {
                cf = 1.0 + ytm/freq;
            } else {
                cf = ytm/freq;
            }

            if (ti/dt-(ti/dt).floor()).abs() < 1e-7 {
                df = f64::powf(1.0+ytm/freq, -ti*freq);
            } else {
                df = 1.0/(1.0+ytm*ti);
            }

            p += cf*df;
            ti -= dt;
        }
        p
    }
}

pub fn smith_wilson_ytm(ltfr: f64, alpha: f64, tenor: Vec<f64>, ytm: Vec<f64>, freq: f64) -> TermStructure {
    let ltfr2 = (1.0+ltfr).ln();
    let n = tenor.len();
    let mut n2: usize;
    let mut t: Vec<f64>;
    let mut c: Vec<Vec<f64>>;

    // 1. C
    if freq == 0.0 {
        n2 = n;
        c = vec![vec![0.0; n2]; n];
        t = vec![0.0; n2];
        for i in 0..n2 {
            t[i] = tenor[i];
            for j in 0..n {
                if i == j { c[i][j] = 1.0; }
            }
        }
    } else {
        let mut l = 0;
        for i in 0..n {
            l += (tenor[i]*freq).ceil() as usize;
        }
    
        let mut c_col_candidate = vec![0.0; l];
        let mut c_col_candidate2 = vec![0.0; l];
    
        let mut k = 0;
        for i in 0..n {
            for j in 0..((tenor[i]*freq).ceil() as usize) {
                c_col_candidate[k] = tenor[i] - (j as f64)/freq;
                k += 1;
            }
        }

        n2 = 0;
        let mut tmp: f64;
        for _ in 0..l {
            tmp = *c_col_candidate.iter().min_by(|a, b| {a.partial_cmp(b).unwrap()}).unwrap();
            if tmp == 99999.0 { break; }
            c_col_candidate2[n2] = tmp;
            n2 += 1;
            for j in 0..l {
                if c_col_candidate[j] == tmp { c_col_candidate[j] = 99999.0; }
            }
        }
        t = Vec::from(&c_col_candidate2[..n2]);
        c = vec![vec![0.0; n2]; n];

        for i in 0..n {
            tmp = tenor[i];
            for j in 0..n2 {
                if t[j] > tmp {
                    c[i][j] = 0.0;
                } else if t[j] == tmp {
                    c[i][j] = 1.0 + ytm[i]/freq;
                } else if ((12.0*(tmp-t[j])) as i32) % (12/(freq as i32)) == 0 {
                    c[i][j] = ytm[i]/freq;
                } else {
                    c[i][j] = 0.0;
                }
            }
        }
    }

    // 2. m, u
    let mut m = vec![0.0; n];
    let mut u = vec![0.0; n2];

    for i in 0..n {
        m[i] = ytm_price(t[i], ytm[i], freq);
    }
    for i in 0..n2 {
        u[i] = (-ltfr2*t[i]).exp();
    }
    
    // 3. W
    let mut w = vec![vec![0.0; n2]; n2];
    for i in 0..n2 {
        for j in i..n2 {
            w[i][j] = (-ltfr2*(t[i]+t[j])).exp()*(alpha*t[i].min(t[j])-(-alpha*t[i].max(t[j])).exp()*(alpha*t[i].min(t[j])).sinh());
            if i != j { w[j][i] = w[i][j]; }
        }
    }
    
    // 4. zeta
    let mut m_cu = c.map(&u);
    for i in 0..n {
        m_cu[i] = m[i] - m_cu[i];
    }
    let zeta = (c.mul(&w).mul(&c.tp())).inv().map(&m_cu);
    let zeta2 = c.tp().map(&zeta);

    // output
    let mut p = [0.0; 241];

    for i in 0..=240 {
        let tt = (i as f64)/12.0;
        p[i] = (-ltfr2*tt).exp();
        for j in 0..n2 {
            p[i] += zeta2[j]*(-ltfr2*(tt+t[j])).exp()*(alpha*tt.min(t[j])-(-alpha*tt.max(t[j])).exp()*(alpha*tt.min(t[j])).sinh());
        }
    }

    TermStructure { p: p, f: [0.0; 241] }
}


trait Matrix {
    fn mul(&self, _rhs: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn tp(&self) -> Vec<Vec<f64>>;
    fn map(&self, _rhs: &Vec<f64>) -> Vec<f64>;
    fn chol(&self) -> Vec<Vec<f64>>;
    fn inv(&self) -> Vec<Vec<f64>>;
}

impl Matrix for Vec<Vec<f64>> {
    fn inv(&self) -> Vec<Vec<f64>> {
        let l = self.chol();
        let n = l.len();
        let mut l_inv = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in j..n {
                if i==j {
                    l_inv[i][j] = 1.0/l[i][i];
                } else {
                    for k in j..i {
                        l_inv[i][j] -= l[i][k]*l_inv[k][j];
                    }
                    l_inv[i][j] /= l[i][i];
                }
            }
        }
        l_inv.tp().mul(&l_inv)
    }
    fn chol(&self) -> Vec<Vec<f64>> {
        let n = self.len();
        let m = self[0].len();
        if n != m { panic!("n != m"); }
        let mut l = vec![vec![0.0; n]; n];
        let mut sum: f64;
        for i in 0..n {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..j {
                    sum += l[i][k]*l[j][k];
                }
                if i==j {
                    l[i][j] = (self[i][i]-sum).sqrt();
                } else {
                    l[i][j] = 1.0/l[j][j]*(self[i][j]-sum);
                }
            }
        }
        l
    }
    fn map(&self, _rhs: &Vec<f64>) -> Vec<f64> {
        let n = self.len();
        let m = self[0].len();
        let mut v = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                v[i] += self[i][j]*_rhs[j];
            }
        }
        v
    }
    fn mul(&self, _rhs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = self.len();
        let l = self[0].len();
        let m = _rhs[0].len();
        let mut b = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                for k in 0..l {
                    b[i][j] += self[i][k]*_rhs[k][j];
                }
            }
        }
        b
    }
    fn tp(&self) -> Vec<Vec<f64>> {
        let n = self.len();
        let m = self[0].len();
        let mut b = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                b[i][j] = self[j][i];
            }
        }
        b
    }
}
