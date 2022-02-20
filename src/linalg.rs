#[derive(Debug)]
pub struct Vector {
    data: Vec<f64>
}

impl Vector {
    /// **신규 벡터 생성**
    /// 
    /// 신규 벡터를 생성하는 메서드입니다
    /// 
    /// # Examples
    /// 
    /// ```
    /// use esgrs::linalg::*;
    /// 
    /// let u: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
    /// ```
    pub fn new(v: Vec<f64>) -> Vector {
        Vector { data: v }
    }
}

/// **동일성 연산**
/// 
/// 두 벡터의 동일성 연산(==) 메서드입니다.__rust_force_expr!
/// 
/// # Examples
/// 
/// ```
/// use esgrs::linalg::*;
/// 
/// let u: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
/// let v: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
/// assert_eq!(u, v);
/// ```
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