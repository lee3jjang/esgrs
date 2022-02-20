fn erf(z: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p*z.abs());

    //Direct calculation using formula 7.1.26 is absolutely correct
    //But calculation of nth order polynomial takes O(n^2) operations
    // return 1 - (a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t) * Math.Exp(-1 * x * x);

    //Horner's method, takes O(n) operations for nth order polynomial
    return 1.0 - ((((((a5 * t + a4) * t) + a3) * t + a2) * t) + a1) * t * (-z*z).exp();
}

pub fn norm_cdf(z: f64) -> f64 {
    let sign: f64 = if z < 0.0 { -1.0 } else { 1.0 };
    return 0.5*(1.0+sign*erf(z/f64::sqrt(2.0)));
}

pub fn norm_pdf(z: f64) -> f64 {
    return (-z*z/2.0).exp()/(2.0*std::f64::consts::PI).sqrt();
}