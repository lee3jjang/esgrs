pub trait UniFn {
    
    fn value(&self, x: f64) -> f64;

    fn gss(&self, lb: f64, ub: f64) -> f64 {
        let tol: f64 = 1e-12;
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