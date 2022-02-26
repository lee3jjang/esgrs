#[deprecated(since="0.2.0", note="please use `DiffUniFn` instead")]
pub trait UniFn {
    
    fn value(&self, x: f64) -> f64;

    fn gss(&self, lb: f64, ub: f64) -> f64 {
        let tol: f64 = 1e-15;
        let gr: f64 = (1.0 + f64::sqrt(5.0))/2.0;
        let mut a: f64 = lb;
        let mut b: f64 = ub;

        let mut c = b-(b-a)/gr;
        let mut d = a+(b-a)/gr;
        while (b-a).abs() > tol {
            if self.value(c) < self.value(d) {
                b = d;
            } else {
                a = c;
            }
            c = b-(b-a)/gr;
            d = a+(b-a)/gr;
        }
        return (b+a)/2.0;
    }

}

pub trait DiffUniFn {
    
    fn value(&self, x: f64) -> f64;

    fn deriv(&self, x: f64) -> f64;

    fn newton_raphson(&self, x0: f64) -> f64 {
        let tol = 1e-15;
        let mut x = x0;
        loop {
            let y = self.value(x);
            if y < tol { break; }
            x -= y/self.deriv(x);
        }
        return x;
    }

}