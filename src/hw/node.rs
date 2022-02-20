use crate::optim::UniFn;
use crate::stats::norm_cdf;

#[derive(Debug, Copy, Clone)]
pub struct TermStructure {
    pub p: [f64; 241],
    pub f: [f64; 241],
    pub df: [f64; 241],
}

/// **B 노드**
/// 
/// B(a; t1, t2)를 계산하는 노드를 생성합니다.
/// 
/// # Examples
/// 
/// ```
/// use esgrs::hw::node::{B};
/// 
/// let a = 0.1;
/// let mut b100_125 = B::new(1.00, 1.25);
/// b100_125.forward(a);
/// b100_125.backward(dout);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct B {
    alpha: f64,
    t1: f64,
    t2: f64,
}

impl B {
    pub fn new(t1: f64, t2: f64) -> Self {
        return B { alpha: f64::NAN, t1: t1, t2: t2 };
    }
    pub fn forward(&mut self, alpha: f64) -> f64 {
        self.alpha = alpha;
        let out = (1.0-(-alpha*(self.t2-self.t1)).exp())/alpha;
        return out;
    }
    pub fn backward(&self, dout: f64) -> f64 {
        let dalpha = -(1.0-(-self.alpha*(self.t2-self.t1)).exp())/(self.alpha*self.alpha)
            + (-self.alpha*(self.t2-self.t1)).exp()/self.alpha*(self.t2-self.t1)*dout;
        return dalpha;
    }
}

/// **Vr 노드**
/// 
/// Vr(a, sigma; t)를 계산하는 노드를 생성합니다.
/// 
/// # Examples
/// 
/// ```
/// use esgrs::hw::node::{Vr};
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Vr {
    alpha: f64,
    sigma1: f64,
    sigma2: f64,
    sigma3: f64,
    sigma5: f64,
    sigma7: f64,
    sigma10: f64,
    t: f64,
}

impl Vr {
    pub fn new(t: f64) -> Self {
        return Vr { alpha: f64::NAN, sigma1: f64::NAN, sigma2: f64::NAN, sigma3: f64::NAN, sigma5: f64::NAN, sigma7: f64::NAN, sigma10: f64::NAN, t: t };
    }
    pub fn forward(&self, alpha: f64, sigma1: f64, sigma2: f64, sigma3: f64, sigma5: f64, sigma7: f64, sigma10: f64) -> f64 {
        let out = if self.t == 1.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*0.0).exp())
        } else if self.t == 2.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*1.0).exp()-(2.0*alpha*0.0).exp())
            + sigma2*sigma2/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*1.0).exp())
        } else if self.t == 3.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*1.0).exp()-(2.0*alpha*0.0).exp())
            + sigma2*sigma2/2.0/alpha*((2.0*alpha*2.0).exp()-(2.0*alpha*1.0).exp())
            + sigma3*sigma3/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*2.0).exp())
        } else if self.t == 5.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*1.0).exp()-(2.0*alpha*0.0).exp())
            + sigma2*sigma2/2.0/alpha*((2.0*alpha*2.0).exp()-(2.0*alpha*1.0).exp())
            + sigma3*sigma3/2.0/alpha*((2.0*alpha*3.0).exp()-(2.0*alpha*2.0).exp())
            + sigma5*sigma5/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*3.0).exp())
        } else if self.t == 7.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*1.0).exp()-(2.0*alpha*0.0).exp())
            + sigma2*sigma2/2.0/alpha*((2.0*alpha*2.0).exp()-(2.0*alpha*1.0).exp())
            + sigma3*sigma3/2.0/alpha*((2.0*alpha*3.0).exp()-(2.0*alpha*2.0).exp())
            + sigma5*sigma5/2.0/alpha*((2.0*alpha*5.0).exp()-(2.0*alpha*3.0).exp())
            + sigma7*sigma7/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*5.0).exp())
        } else if self.t == 10.0 {
            sigma1*sigma1/2.0/alpha*((2.0*alpha*1.0).exp()-(2.0*alpha*0.0).exp())
            + sigma2*sigma2/2.0/alpha*((2.0*alpha*2.0).exp()-(2.0*alpha*1.0).exp())
            + sigma3*sigma3/2.0/alpha*((2.0*alpha*3.0).exp()-(2.0*alpha*2.0).exp())
            + sigma5*sigma5/2.0/alpha*((2.0*alpha*5.0).exp()-(2.0*alpha*3.0).exp())
            + sigma7*sigma7/2.0/alpha*((2.0*alpha*7.0).exp()-(2.0*alpha*5.0).exp())
            + sigma10*sigma10/2.0/alpha*((2.0*alpha*self.t).exp()-(2.0*alpha*7.0).exp())
        } else {
            panic!("Vr(t) domain error");
        };
        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
        let (dalpha, dsigma1, dsigma2, dsigma3, dsigma5, dsigma7, dsigma10) = if self.t == 1.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = 0.0;
            let dsigma3 = 0.0;
            let dsigma5 = 0.0;
            let dsigma7 = 0.0;
            let dsigma10 = 0.0;
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else if self.t == 2.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(1.0*(2.0*1.0*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp())
            -self.sigma2*self.sigma2/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*1.0).exp()) + self.sigma2*self.sigma2/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-1.0*(2.0*1.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = self.sigma2/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*1.0).exp());
            let dsigma3 = 0.0;
            let dsigma5 = 0.0;
            let dsigma7 = 0.0;
            let dsigma10 = 0.0;
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else if self.t == 3.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(1.0*(2.0*1.0*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp())
            -self.sigma2*self.sigma2/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp()) + self.sigma2*self.sigma2/self.alpha*(2.0*(2.0*2.0*self.alpha).exp()-1.0*(2.0*1.0*self.alpha).exp())
            -self.sigma3*self.sigma3/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*2.0).exp()) + self.sigma3*self.sigma3/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-2.0*(2.0*2.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = self.sigma2/self.alpha*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp());
            let dsigma3 = self.sigma3/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*2.0).exp());
            let dsigma5 = 0.0;
            let dsigma7 = 0.0;
            let dsigma10 = 0.0;
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else if self.t == 5.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(1.0*(2.0*1.0*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp())
            -self.sigma2*self.sigma2/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp()) + self.sigma2*self.sigma2/self.alpha*(2.0*(2.0*2.0*self.alpha).exp()-1.0*(2.0*1.0*self.alpha).exp())
            -self.sigma3*self.sigma3/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp()) + self.sigma3*self.sigma3/self.alpha*(3.0*(2.0*3.0*self.alpha).exp()-2.0*(2.0*2.0*self.alpha).exp())
            -self.sigma5*self.sigma5/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*3.0).exp()) + self.sigma5*self.sigma5/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-3.0*(2.0*3.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = self.sigma2/self.alpha*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp());
            let dsigma3 = self.sigma3/self.alpha*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp());
            let dsigma5 = self.sigma5/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*3.0).exp());
            let dsigma7 = 0.0;
            let dsigma10 = 0.0;
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else if self.t == 7.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(1.0*(2.0*1.0*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp())
            -self.sigma2*self.sigma2/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp()) + self.sigma2*self.sigma2/self.alpha*(2.0*(2.0*2.0*self.alpha).exp()-1.0*(2.0*1.0*self.alpha).exp())
            -self.sigma3*self.sigma3/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp()) + self.sigma3*self.sigma3/self.alpha*(3.0*(2.0*3.0*self.alpha).exp()-2.0*(2.0*2.0*self.alpha).exp())
            -self.sigma5*self.sigma5/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*5.0).exp()-(2.0*self.alpha*3.0).exp()) + self.sigma5*self.sigma5/self.alpha*(5.0*(2.0*5.0*self.alpha).exp()-3.0*(2.0*3.0*self.alpha).exp())
            -self.sigma7*self.sigma7/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*5.0).exp()) + self.sigma7*self.sigma7/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-5.0*(2.0*5.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = self.sigma2/self.alpha*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp());
            let dsigma3 = self.sigma3/self.alpha*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp());
            let dsigma5 = self.sigma5/self.alpha*((2.0*self.alpha*5.0).exp()-(2.0*self.alpha*3.0).exp());
            let dsigma7 = self.sigma7/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*5.0).exp());
            let dsigma10 = 0.0;
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else if self.t == 10.0 {
            let dalpha = -self.sigma1*self.sigma1/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp()) + self.sigma1*self.sigma1/self.alpha*(1.0*(2.0*1.0*self.alpha).exp()-0.0*(2.0*0.0*self.alpha).exp())
            -self.sigma2*self.sigma2/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp()) + self.sigma2*self.sigma2/self.alpha*(2.0*(2.0*2.0*self.alpha).exp()-1.0*(2.0*1.0*self.alpha).exp())
            -self.sigma3*self.sigma3/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp()) + self.sigma3*self.sigma3/self.alpha*(3.0*(2.0*3.0*self.alpha).exp()-2.0*(2.0*2.0*self.alpha).exp())
            -self.sigma5*self.sigma5/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*5.0).exp()-(2.0*self.alpha*3.0).exp()) + self.sigma5*self.sigma5/self.alpha*(5.0*(2.0*5.0*self.alpha).exp()-3.0*(2.0*3.0*self.alpha).exp())
            -self.sigma7*self.sigma7/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*7.0).exp()-(2.0*self.alpha*5.0).exp()) + self.sigma7*self.sigma7/self.alpha*(7.0*(2.0*7.0*self.alpha).exp()-5.0*(2.0*5.0*self.alpha).exp())
            -self.sigma10*self.sigma10/(2.0*self.alpha*self.alpha)*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*7.0).exp()) + self.sigma10*self.sigma10/self.alpha*(self.t*(2.0*self.t*self.alpha).exp()-7.0*(2.0*7.0*self.alpha).exp());
            let dsigma1 = self.sigma1/self.alpha*((2.0*self.alpha*1.0).exp()-(2.0*self.alpha*0.0).exp());
            let dsigma2 = self.sigma2/self.alpha*((2.0*self.alpha*2.0).exp()-(2.0*self.alpha*1.0).exp());
            let dsigma3 = self.sigma3/self.alpha*((2.0*self.alpha*3.0).exp()-(2.0*self.alpha*2.0).exp());
            let dsigma5 = self.sigma5/self.alpha*((2.0*self.alpha*5.0).exp()-(2.0*self.alpha*3.0).exp());
            let dsigma7 = self.sigma7/self.alpha*((2.0*self.alpha*7.0).exp()-(2.0*self.alpha*5.0).exp());
            let dsigma10 = self.sigma10/self.alpha*((2.0*self.alpha*self.t).exp()-(2.0*self.alpha*7.0).exp());
            (dalpha*dout, dsigma1*dout, dsigma2*dout, dsigma3*dout, dsigma5*dout, dsigma7*dout, dsigma10*dout)
        } else {
            panic!("Vr(t) domain error");
        };

        return (dalpha, dsigma1, dsigma2, dsigma3, dsigma5, dsigma7, dsigma10);
    }
}

/// **A 노드**
/// 
/// A(b, v_r; t1, t2)를 계산하는 노드를 생성합니다.
/// 
/// # Examples
/// 
/// ```
/// use esgrs::hw::node::{Vr};
/// ```
#[derive(Debug, Copy, Clone)]
pub struct A {
    b: f64,
    vr: f64,
    ln_p_t2_p_t1: f64,
    f: f64,
}

impl A {
    pub fn new(t1: f64, t2: f64, ts: TermStructure) -> Self {
        let ln_p_t2_p_t1 = (ts.p[(12.0*t2) as usize]/ts.p[(12.0*t1) as usize]).ln();
        let f = ts.f[(12.0*t1) as usize];
        return A { b: f64::NAN, vr: f64::NAN, ln_p_t2_p_t1: ln_p_t2_p_t1, f: f };
    }
    pub fn forward(&mut self, b: f64, vr: f64) -> f64 {
        self.b = b;
        self.vr = vr;
        let out = self.ln_p_t2_p_t1 + self.b*self.f - 0.5*self.b*self.b*self.vr;
        return out;
    }
    pub fn backward(&self, dout: f64) -> (f64, f64) {
        let db = (self.f - self.b*self.vr)*dout;
        let dvr = -0.5*self.b*self.b*dout;
        return (db, dvr);
    }
}

/// **Vp 노드**
/// 
/// Vp(b, v_r)를 계산하는 노드를 생성합니다.
/// 
/// # Examples
/// 
/// ```
/// use esgrs::hw::node::{Vr};
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Vp {
    b: f64,
    vr: f64,
}

impl Vp {
    pub fn new() -> Self {
        Vp { b: f64::NAN, vr: f64::NAN }
    }
    pub fn forward(&mut self, b:f64, vr: f64) -> f64 {
        self.b = b;
        self.vr = vr;
        let out = vr*b*b;
        return out;
    }
    pub fn backward(&self, dout: f64) -> (f64, f64) {
        let dvr = self.b*self.b*dout;
        let db = self.vr*dout;
        return (db, dvr);
    }
}


#[derive(Debug, Copy, Clone)]
pub struct RstarT1 {
    a: [f64; 4],
    b: [f64; 4],
    k: f64,
    out: f64,
}

impl RstarT1 {
    pub fn new(k: f64) -> Self {
        RstarT1 { a: [f64::NAN; 4], b: [f64::NAN; 4], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 4], b: [f64; 4]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT1 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 4], [f64; 4]) {
        let da = [1.0; 4];
        let db = [1.0; 4];
        return (da, db);
    }
}

pub struct JamshidianT1 {
    a: [f64; 4],
    b: [f64; 4],
    k: f64,
}

impl UniFn for JamshidianT1 {
    fn value(&self, r: f64) -> f64 {
        let num = 4;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RstarT2 {
    a: [f64; 8],
    b: [f64; 8],
    k: f64,
    out: f64,
}

impl RstarT2 {
    pub fn new(k: f64) -> Self {
        RstarT2 { a: [f64::NAN; 8], b: [f64::NAN; 8], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 8], b: [f64; 8]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT2 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 8], [f64; 8]) {
        let da = [1.0; 8];
        let db = [1.0; 8];
        return (da, db);
    }
}

pub struct JamshidianT2 {
    a: [f64; 8],
    b: [f64; 8],
    k: f64,
}

impl UniFn for JamshidianT2 {
    fn value(&self, r: f64) -> f64 {
        let num = 8;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RstarT3 {
    a: [f64; 12],
    b: [f64; 12],
    k: f64,
    out: f64,
}

impl RstarT3 {
    pub fn new(k: f64) -> Self {
        RstarT3 { a: [f64::NAN; 12], b: [f64::NAN; 12], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 12], b: [f64; 12]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT3 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 12], [f64; 12]) {
        let da = [1.0; 12];
        let db = [1.0; 12];
        return (da, db);
    }
}
    
pub struct JamshidianT3 {
    a: [f64; 12],
    b: [f64; 12],
    k: f64,
}

impl UniFn for JamshidianT3 {
    fn value(&self, r: f64) -> f64 {
        let num = 12;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RstarT5 {
    a: [f64; 20],
    b: [f64; 20],
    k: f64,
    out: f64,
}

impl RstarT5 {
    pub fn new(k: f64) -> Self {
        RstarT5 { a: [f64::NAN; 20], b: [f64::NAN; 20], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 20], b: [f64; 20]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT5 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 20], [f64; 20]) {
        let da = [1.0; 20];
        let db = [1.0; 20];
        return (da, db);
    }
}

pub struct JamshidianT5 {
    a: [f64; 20],
    b: [f64; 20],
    k: f64,
}

impl UniFn for JamshidianT5 {
    fn value(&self, r: f64) -> f64 {
        let num = 20;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RstarT7 {
    a: [f64; 28],
    b: [f64; 28],
    k: f64,
    out: f64,
}

impl RstarT7 {
    pub fn new(k: f64) -> Self {
        RstarT7 { a: [f64::NAN; 28], b: [f64::NAN; 28], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 28], b: [f64; 28]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT7 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 28], [f64; 28]) {
        let da = [1.0; 28];
        let db = [1.0; 28];
        return (da, db);
    }
}

pub struct JamshidianT7 {
    a: [f64; 28],
    b: [f64; 28],
    k: f64,
}

impl UniFn for JamshidianT7 {
    fn value(&self, r: f64) -> f64 {
        let num = 28;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RstarT10 {
    a: [f64; 40],
    b: [f64; 40],
    k: f64,
    out: f64,
}

impl RstarT10 {
    pub fn new(k: f64) -> Self {
        RstarT10 { a: [f64::NAN; 40], b: [f64::NAN; 40], k: k, out: f64::NAN }
    }
    pub fn forward(&mut self, a: [f64; 40], b: [f64; 40]) -> f64 {
        self.a = a;
        self.b = b;
        self.out = JamshidianT10 { a: self.a, b: self.b, k: self.k }.gss(0.0, 1.0);
        return self.out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 40], [f64; 40]) {
        let da = [1.0; 40];
        let db = [1.0; 40];
        return (da, db);
    }
}

pub struct JamshidianT10 {
    a: [f64; 40],
    b: [f64; 40],
    k: f64,
}

impl UniFn for JamshidianT10 {
    fn value(&self, r: f64) -> f64 {
        let num = 40;
        let mut eq: f64 = 0.0;
        for i in 0..num {
            let ci = if i != num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            eq += ci*(self.a[i]-self.b[i]*r).exp();
        }
        return (eq-1.0).abs();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT1 {
    a: [f64; 4],
    b: [f64; 4],
    vp: [f64; 4],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 4],
}

impl PSwaptionT1 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 4;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 4];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT1 { a: [f64::NAN; 4], b: [f64::NAN; 4], vp: [f64::NAN; 4], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 4], b: [f64; 4], vp: [f64; 4], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 4;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 4], [f64; 4], [f64; 4], f64) {
        let da = [1.0; 4];
        let db = [1.0; 4];
        let dvp = [1.0; 4];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT2 {
    a: [f64; 8],
    b: [f64; 8],
    vp: [f64; 8],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 8],
}

impl PSwaptionT2 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 8;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 8];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT2 { a: [f64::NAN; 8], b: [f64::NAN; 8], vp: [f64::NAN; 8], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 8], b: [f64; 8], vp: [f64; 8], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 8;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 8], [f64; 8], [f64; 8], f64) {
        let da = [1.0; 8];
        let db = [1.0; 8];
        let dvp = [1.0; 8];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT3 {
    a: [f64; 12],
    b: [f64; 12],
    vp: [f64; 12],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 12],
}

impl PSwaptionT3 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 12;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 12];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT3 { a: [f64::NAN; 12], b: [f64::NAN; 12], vp: [f64::NAN; 12], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 12], b: [f64; 12], vp: [f64; 12], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 12;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 12], [f64; 12], [f64; 12], f64) {
        let da = [1.0; 12];
        let db = [1.0; 12];
        let dvp = [1.0; 12];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT5 {
    a: [f64; 20],
    b: [f64; 20],
    vp: [f64; 20],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 20],
}

impl PSwaptionT5 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 20;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 20];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT5 { a: [f64::NAN; 20], b: [f64::NAN; 20], vp: [f64::NAN; 20], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 20], b: [f64; 20], vp: [f64; 20], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 20;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 20], [f64; 20], [f64; 20], f64) {
        let da = [1.0; 20];
        let db = [1.0; 20];
        let dvp = [1.0; 20];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT7 {
    a: [f64; 28],
    b: [f64; 28],
    vp: [f64; 28],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 28],
}

impl PSwaptionT7 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 28;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 28];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT7 { a: [f64::NAN; 28], b: [f64::NAN; 28], vp: [f64::NAN; 28], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 28], b: [f64; 28], vp: [f64; 28], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 28;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 28], [f64; 28], [f64; 28], f64) {
        let da = [1.0; 28];
        let db = [1.0; 28];
        let dvp = [1.0; 28];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PSwaptionT10 {
    a: [f64; 40],
    b: [f64; 40],
    vp: [f64; 40],
    rstar: f64,
    k: f64,
    p_tf: f64,
    p_ti: [f64; 40],
}

impl PSwaptionT10 {
    pub fn new(t: f64, k: f64, ts: TermStructure) -> Self {
        let num = 40;
        let p_tf = ts.p[(12.0*t) as usize];
        let mut p_ti = [0.0; 40];
        for i in 0..num {
            p_ti[i] = ts.p[(12.0*t) as usize + i + 1];
        } 
        PSwaptionT10 { a: [f64::NAN; 40], b: [f64::NAN; 40], vp: [f64::NAN; 40], rstar: f64::NAN, k: k, p_tf: p_tf, p_ti: p_ti }
    }
    pub fn forward(&mut self, a: [f64; 40], b: [f64; 40], vp: [f64; 40], rstar: f64) -> f64 {
        self.a = a;
        self.b = b;
        self.vp = vp;
        self.rstar = rstar;

        let mut out = 0.0;
        let num = 40;
        for i in 0..num {
            let xi = (a[i]-b[i]*rstar).exp();
            let ci = if i == num-1 { 0.25*self.k } else { 1.0+0.25*self.k };
            let dplusi = (self.p_tf/self.p_ti[i]*xi).ln()/vp[i].sqrt() + 0.5*vp[i].sqrt();
            let dminusi = dplusi - vp[i].sqrt();
            let zbp = xi*self.p_tf*norm_cdf(dplusi) - self.p_ti[i]*norm_cdf(dminusi);
            out += ci*zbp;
        }

        return out;
    }
    #[allow(dead_code)]
    pub fn backward(&self, dout: f64) -> ([f64; 40], [f64; 40], [f64; 40], f64) {
        let da = [1.0; 40];
        let db = [1.0; 40];
        let dvp = [1.0; 40];
        let drstar = 1.0;
        return (da, db, dvp, drstar);
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MRSE {
    pswaption: [[f64; 6]; 6],
    pswaption_mkt: [[f64; 6]; 6],
}

impl MRSE {
    pub fn new(pswaption_mkt: [[f64; 6]; 6]) -> Self {
        MRSE { pswaption: [[f64::NAN; 6]; 6], pswaption_mkt: pswaption_mkt }
    }
    pub fn forward(&mut self, pswaption: [[f64; 6]; 6]) -> f64 {
        self.pswaption = pswaption;

        let mut out = 0.0;
        for i in 0..6 {
            for j in 0..6 {
                out += (pswaption[i][j] - self.pswaption_mkt[i][j])*(pswaption[i][j] - self.pswaption_mkt[i][j])/(self.pswaption_mkt[i][j]*self.pswaption_mkt[i][j]);
            }
        }
        out /= 36.0;

        return out;
    }
    pub fn backward(&self, dout: f64) -> [[f64; 6]; 6] {
        let mut dpswaption = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                dpswaption[i][j] = -2.0*(1.0-self.pswaption[i][j]/self.pswaption_mkt[i][j])/self.pswaption_mkt[i][j]/36.0;
            }
        }
        return dpswaption;
    }
}