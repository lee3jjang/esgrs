use crate::linalg::Vector;

#[test]
fn test_vector_equality() {
    let u: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
    let v: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
    let w: Vector = Vector::new(vec![-6.0, 7.0, -99.0]);
    assert_eq!(u, v);
    assert_ne!(u, w);
}

#[test]
fn test_different_dimension_vectors_equality() {
    let u: Vector = Vector::new(vec![1.0, 3.0, 2.0]);
    let v: Vector = Vector::new(vec![1.0, 3.0, 2.0, 4.0]);
    assert_ne!(u, v);
}

// #[test]
// fn test_matrix_add() {

//     let u = vec![1.0, 3.0, 2.0];
//     let v = vec![-6.0, 7.0, -99.0];
//     let w = vec![-5.0, 4.0, -97.0];
    
//     assert_eq!(u+v, w);
// }