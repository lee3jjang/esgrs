pub fn inv(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let l = chol(a);
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
    mul(&tp(&l_inv), &l_inv)
}
pub fn chol(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
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
                l[i][j] = (a[i][i]-sum).sqrt();
            } else {
                l[i][j] = 1.0/l[j][j]*(a[i][j]-sum);
            }
        }
    }
    l
}
pub fn map(a: &Vec<Vec<f64>>, v: &Vec<f64>) -> Vec<f64> {
    let n = a.len();
    let m = a[0].len();
    let mut w = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            w[i] += a[i][j]*v[j];
        }
    }
    w
}
pub fn mul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let l = a[0].len();
    let m = b[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..l {
                c[i][j] += a[i][k]*b[k][j];
            }
        }
    }
    c
}
pub fn tp(a: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut b = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            b[i][j] = a[j][i];
        }
    }
    b
}