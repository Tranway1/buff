use packed_simd::i32x4;

#[test]
fn test_simd() {
    let a = i32x4::new(1, 1, 3, 3);
    let b = i32x4::new(2, 2, 0, 0);

// ge: >= (Greater Eequal; see also lt, le, gt, eq, ne).
    let m = a.ge(i32x4::splat(2));

    if m.any() {
        // all / any / none allow coherent control flow
        let d = m.select(a, b);
        assert_eq!(d, i32x4::new(2, 2, 3, 3));
    }
}