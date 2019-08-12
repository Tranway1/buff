use queues::*;
use crate::stats::Stats;
use std::time::SystemTime;
use num::Num;

#[derive(Default,Clone)]
pub struct Tree<T>
    where T:Num + Default{
    root: Stats<T>,
    left: Option<Box<Tree<T>>>,
    right: Option<Box<Tree<T>>>,
}
impl<T> Tree<T>
    where T: Num + Default
    {
        fn new(root: Stats<T>) -> Tree<T> {
            Tree {
                root: root,
                ..Default::default()
            }
        }
        fn left(mut self, leaf: Tree<T>) -> Self {
            self.left = Some(Box::new(leaf));
            self
        }
        fn right(mut self, leaf: Tree<T>) -> Self {
            self.right = Some(Box::new(leaf));
            self
        }

        fn get_left(&self) -> &Option<Box<Tree<T>>>{
            &self.left
        }

        fn get_right(&self) -> &Option<Box<Tree<T>>>{
            &self.right
        }

        fn get_root(&self) -> &Stats<T>{
            &self.root
        }
    }


#[test]
fn test_stats_tree(){
    let mytree = Tree::new( Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 1, 1, 14))
        .left(
            Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 2, 1, 14))
                .right(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 3, 1, 14)))
        )
        .right(
            Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 4, 1, 14))
                .left(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 5, 1, 14)))
                .right(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 6, 1, 14)))
        );
    println!("count of rount: {}", mytree.root.get_count());
    let left = mytree.get_left();
    match left {
        Some(ref l) => println!("count of rount: {}", l.get_root().get_count()),
        None => println!("has no value"),
    }
}


#[test]
fn test_build_stats_tree(){
    // Create a simple Queue
    let mut q: Queue<Tree<i32>> = queue![];

    // Add some elements to it
    q.add(Tree::new( Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 1, 1, 14)));
    q.add(Tree::new( Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 2, 1, 14)));
    q.add(Tree::new( Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 3, 1, 14)));

    // Check the Queue's size
    q.size();  // 3

    // Remove an element
    q.remove();  // Ok(1)

    // Check the Queue's size
    q.size();  // 2

    // Peek at the next element scheduled for removal
    q.peek();  // Ok(-2)

    // Confirm that the Queue size hasn't changed
    q.size();  // 2

    // Remove the remaining elements
    q.remove();  // Ok(-2)


    q.remove();  // Ok(3)

    // Peek into an empty Queue
    q.peek();  // Raises an error

    // Attempt to remove an element from an empty Queue
    q.remove();  // Raises an error

    let mytree = Tree::new( Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 1, 1, 14))
        .left(
            Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 2, 1, 14))
                .right(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 3, 1, 14)))
        )
        .right(
            Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 4, 1, 14))
                .left(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 5, 1, 14)))
                .right(Tree::new(Stats::new(SystemTime::now(), SystemTime::now(), 3, 2, 6, 1, 14)))
        );
    println!("count of rount: {}", mytree.root.get_count());
    let left = mytree.get_left();
    match left {
        Some(ref l) => println!("count of rount: {}", l.get_root().get_count()),
        None => println!("has no value"),
    }
}