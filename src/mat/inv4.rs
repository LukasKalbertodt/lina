//! Helper code for 4x4 matrix inversion.
//!
//! While 1x1 and 2x2 matrix inversions are simple and 3x3 inversion is still
//! quite managable, inverting a 4x4 matrix is quite involved. There are lots
//! of algorithms flying around on the internet. Many basically implement
//! "Cramers rule".
//!
//! When implementing this/choosing an algorithm we are interested in two
//! things: (a) speed and (b) numeric stability. The latter means how badly the
//! result is influenced by float rounding errors.
//!
//! Probably the most copied code is old code from the Mesa project. But this is
//! not optimal in terms of speed and numeric stability. The implementation
//! here borrow a lot from [this][1]. It is code generated by some code
//! licensed as MIT, written by "Willnode". As far as I can tell, no one
//! respects any licenses when it comes to matrix inversion anyway. It's
//! interesting to know whether an algorithm can have copyright even.
//!
//! I have no idea how fast or stable this algorithm is. Someone should probably
//! look into this, but from what I can tell, most linear algebra libraries
//! just use copy&pasted code and don't evaluate these things.
//!
//! [1]: https://stackoverflow.com/a/44446912/2408867
//! [2]: https://github.com/willnode/N-Matrix-Programmer
//!
//! Some related links:
//!
//! - StackOverflow post with several copy&pastable algorithms:
//!   <https://stackoverflow.com/q/1148309/2408867>
//! - Code licensed as CC0, for the non-SSE code using basically the same code
//!   as the old Mesa code:
//!   <https://github.com/niswegmann/small-matrix-inverse>
//! - math.SE post on the topic: <https://math.stackexchange.com/q/473875/340615>
//! - Intel paper with some code:
//!   <https://web.archive.org/web/20131215123403/ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf>
//! - Some paper on how to optimize 4x4 inversion:
//!   <https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf>
//!

use crate::{Matrix, Float};


struct IntermediateResult<F: Float> {
    det: F,
    t2323: F,
    t1323: F,
    t1223: F,
    t0323: F,
    t0223: F,
    t0123: F,
}

fn step_one<F: Float>(m: &Matrix<F, 4, 4>) -> IntermediateResult<F> {
    let t2323 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    let t1323 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    let t1223 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
    let t0323 = m[0][2] * m[3][3] - m[3][2] * m[0][3];
    let t0223 = m[0][2] * m[2][3] - m[2][2] * m[0][3];
    let t0123 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

    let det = F::zero()
        + m[0][0] * (m[1][1] * t2323 - m[2][1] * t1323 + m[3][1] * t1223)
        - m[1][0] * (m[0][1] * t2323 - m[2][1] * t0323 + m[3][1] * t0223)
        + m[2][0] * (m[0][1] * t1323 - m[1][1] * t0323 + m[3][1] * t0123)
        - m[3][0] * (m[0][1] * t1223 - m[1][1] * t0223 + m[2][1] * t0123);

    IntermediateResult { det, t2323, t1323, t1223, t0323, t0223, t0123 }
}

pub(super) fn det<F: Float>(m: &Matrix<F, 4, 4>) -> F {
    let r = step_one(m);
    r.det
}

pub(super) fn inv<F: Float>(m: &Matrix<F, 4, 4>) -> Option<Matrix<F, 4, 4>> {
    let IntermediateResult { det, t2323, t1323, t1223, t0323, t0223, t0123 } = step_one(m);
    if det.is_zero() {
        return None;
    }

    let t2313 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    let t1313 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    let t1213 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
    let t2312 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    let t1312 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    let t1212 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
    let t0313 = m[0][1] * m[3][3] - m[3][1] * m[0][3];
    let t0213 = m[0][1] * m[2][3] - m[2][1] * m[0][3];
    let t0312 = m[0][1] * m[3][2] - m[3][1] * m[0][2];
    let t0212 = m[0][1] * m[2][2] - m[2][1] * m[0][2];
    let t0113 = m[0][1] * m[1][3] - m[1][1] * m[0][3];
    let t0112 = m[0][1] * m[1][2] - m[1][1] * m[0][2];

    let out = Matrix::from_rows([
        [
              (m[1][1] * t2323 - m[2][1] * t1323 + m[3][1] * t1223),
            - (m[1][0] * t2323 - m[2][0] * t1323 + m[3][0] * t1223),
              (m[1][0] * t2313 - m[2][0] * t1313 + m[3][0] * t1213),
            - (m[1][0] * t2312 - m[2][0] * t1312 + m[3][0] * t1212),
        ],
        [
            - (m[0][1] * t2323 - m[2][1] * t0323 + m[3][1] * t0223),
              (m[0][0] * t2323 - m[2][0] * t0323 + m[3][0] * t0223),
            - (m[0][0] * t2313 - m[2][0] * t0313 + m[3][0] * t0213),
              (m[0][0] * t2312 - m[2][0] * t0312 + m[3][0] * t0212),
        ],
        [
              (m[0][1] * t1323 - m[1][1] * t0323 + m[3][1] * t0123),
            - (m[0][0] * t1323 - m[1][0] * t0323 + m[3][0] * t0123),
              (m[0][0] * t1313 - m[1][0] * t0313 + m[3][0] * t0113),
            - (m[0][0] * t1312 - m[1][0] * t0312 + m[3][0] * t0112),
        ],
        [
            - (m[0][1] * t1223 - m[1][1] * t0223 + m[2][1] * t0123),
              (m[0][0] * t1223 - m[1][0] * t0223 + m[2][0] * t0123),
            - (m[0][0] * t1213 - m[1][0] * t0213 + m[2][0] * t0113),
              (m[0][0] * t1212 - m[1][0] * t0212 + m[2][0] * t0112),
        ],
    ]);

    Some(out * det.recip())
}
