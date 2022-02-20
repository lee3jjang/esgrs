use crate::optim::UniFn;
extern crate std;

#[test]
fn test_gss() {
    struct Sin{}
    impl UniFn for Sin {
        fn value(&self, x: f64) -> f64 {
            x.sin()
        }
    }
    
    let x0 = Sin{}.gss(-std::f64::consts::PI, std::f64::consts::PI);
    assert!((x0-(-std::f64::consts::PI/2.0)).abs() < 1e-7);
}