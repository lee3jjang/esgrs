extern crate std;

#[derive(Debug)]
pub struct Vector {
    data: Vec<f64>
}

impl Vector {
    pub fn new(v: Vec<f64>) -> Vector {
        Vector { data: v }
    }
}

impl PartialEq for Vector {
    fn eq(&self, _rhs: &Self) -> bool {
        let n = self.data.len();
        if n != _rhs.data.len() {
            return false;
        }
        for i in 0..n {
            if self.data[i] != _rhs.data[i] {
                return false;
            }
        }
        return true;
    }
}

// impl ops::Add<Vector<T>> for Vector<T> {
//     type Output = Vector<T>;

//     fn add(self, _rhs: Vector<T>) -> Vector<T> {
//         let n = self.len();
//         let mut v = vec![0.0; n];
//         for i in 0..n {
//             v[i] = self[i] + _rhs[i];
//         }
//         v
//     }
// }