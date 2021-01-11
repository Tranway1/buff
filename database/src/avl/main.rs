use time_series_start::avl::set::AvlTreeSet;

pub fn main() {
    let mut set = (1..10_000 as u32).rev().collect::<AvlTreeSet<_>>();
    println!("build tree with length: {}", set.len());
    println!("tree with height: {}", set.get_height());
    for i in 1..10_000 {
        set.take(&i);
    }
    println!("length: {}", set.len());
}