use std::thread::sleep;
use std::borrow::{Borrow, BorrowMut};

static MAX_KEYS : usize = 3;

pub struct BTree<K: PartialOrd + Clone, V: Clone> {
    root: Box<TreeNode<K, V>>
}

impl<K: PartialOrd + Clone + 'static, V: Clone + 'static> BTree<K, V> {
    pub fn new() -> BTree<K, V> {
        BTree { root: Box::new(LeafNode::new()) }
    }

    pub fn insert(mut self, key: K, value: V) -> Self {
        match self.root.insert(key, value){
            Some(res) => {
                let mut new_root = Box::new(InteriorNode::new());
                //let ptr = Box::into_raw(self.root);
                //let old_r = unsafe { Box::from_raw(ptr) };
                new_root.values.push(self.root);
                let child_pivot = res.0;
                let child_node = res.1;
                new_root.keys.push(child_pivot);
                new_root.values.push(child_node);
                self.root = new_root;
                self
            }
            None => {
                self
            }
        }

    }
}


pub struct LeafNode<K: PartialOrd + Clone, V: Clone> {
    keys: Vec<K>,
    values: Vec<V>
}

pub struct InteriorNode<K: PartialOrd + Clone, V: Clone> {
    keys: Vec<K>,
    values: Vec<Box<TreeNode<K, V>>>
}

impl<K: PartialOrd + Clone, V: Clone> InteriorNode<K, V> {
    pub fn new() -> InteriorNode<K, V> {
        InteriorNode {
            keys: Vec::new(),
            values: Vec::new()
        }
    }

    pub fn choose_subtree(&mut self, key: K) -> &mut TreeNode<K, V> {
        for (i,k) in self.keys.iter().enumerate() {
            if *k > key {
                return self.values[i].as_mut()
            }
            else if *k == key {
                return self.values[i+1].as_mut()
            }
            else if i+1 == self.keys.len() {
                return self.values[i+1].as_mut()
            }
        }
        return self.values[0].as_mut()
    }

    pub fn insert_entry(&mut self, key: K, node: Box<TreeNode<K, V>>) {
        if self.keys.len() == 0{
            self.keys.push(key);
            self.values.push(node);
        }
        else {
            let mut inserted = false;
            for (i,k) in self.keys.iter().enumerate() {
                if *k > key {
                    self.keys.insert(i,key.clone());
                    self.values.insert(i, node);
                    inserted = true;
                    break;
                }
                else if *k == key {
                    self.values.insert(i+1, node);
                    inserted = true;
                    break;
                }
                else if i+1 == self.keys.len (){
                    self.keys.push(key);
                    self.values.push(node);
                    break;
                }
            }
        }

        println!("Interior node inserted");
    }

    pub fn insert_node(&mut self, loc: usize , node: Box<TreeNode<K, V>>) {
        self.values.insert(loc,node);
        println!("Interior node inserted");
    }
}

impl<K: PartialOrd + Clone, V: Clone> LeafNode<K, V> {
    pub fn new() -> LeafNode<K, V> {
        LeafNode {
            keys: Vec::new(),
            values: Vec::new()
        }
    }

    pub fn insert_entry(&mut self, key: K, val: V) {
        if self.keys.len() == 0{
            self.keys.push(key);
            self.values.push(val);
        }
        else {
            let mut inserted = false;
            for (i,k) in self.keys.iter().enumerate() {
                if *k > key {
                    self.keys.insert(i,key.clone());
                    self.values.insert(i, val.clone());
                    inserted = true;
                    break;
                }
                else if *k == key {
                    self.values[i] = val.clone();
                    inserted = true;
                    break;
                }
            }
            if inserted == false{
                self.keys.push(key);
                self.values.push(val);
            }
        }
        println!("leaf element inserted");
    }
}

pub trait TreeNode<K: PartialOrd + Clone + 'static, V: Clone + 'static> {
    fn get(&mut self, key: K) -> Option<V>;
    fn split(&mut self) -> (K, Box<TreeNode<K, V>>);
    fn insert(&mut self, key: K, value: V) -> Option<(K, Box<TreeNode<K, V>>)>;
}


impl<K: PartialOrd + Clone + 'static, V: Clone + 'static> TreeNode<K, V> for LeafNode<K, V> {
    fn get(&mut self, key: K) -> Option<V> {
        for (i, k) in self.keys.iter().enumerate() {
            if key == *k {
                return Some(self.values[i].clone())
            }
        }
        None
    }

    fn split(&mut self) -> (K, Box<TreeNode<K, V>>) {
        let mut new_leaf: LeafNode<K, V> = LeafNode::new();
        let pivot: K = self.keys[MAX_KEYS / 2].clone();

        while self.keys.len() > MAX_KEYS / 2 {
            let key = self.keys.pop().unwrap();
            let value = self.values.pop().unwrap();
            new_leaf.insert_entry(key, value);
        }

        (pivot, Box::new(new_leaf))
    }

    fn insert(&mut self, key: K, value: V) -> Option<(K, Box<TreeNode<K, V>>)> {
        self.insert_entry(key, value);
        if self.keys.len() > MAX_KEYS {
            return Some(self.split())
        }
        None
    }
}


impl<K: PartialOrd + Clone + 'static, V: Clone + 'static> TreeNode<K, V> for InteriorNode<K, V> {
    fn get(&mut self, key: K) -> Option<V> {
        let mut subtree = self.choose_subtree(key.clone());
        subtree.get(key)
//        for (i, k) in self.keys.iter().enumerate() {
//            if key == *k {
//                Some(self.values[i].clone())
//            }
//        }
//        None
    }

    fn split(&mut self) -> (K, Box<TreeNode<K, V>>) {
        let mut new_node: InteriorNode<K, V> = InteriorNode::new();
        let pivot: K = self.keys[MAX_KEYS / 2].clone();

        while self.keys.len() > MAX_KEYS / 2 {
            let key = self.keys.pop().unwrap();
            let value = self.values.pop().unwrap();
            new_node.insert_entry(key, value);
        }

        self.keys.pop();/* pop up pivot*/
        new_node.insert_node(0,self.values.pop().unwrap()); /* adjust left-most branch*/
        (pivot, Box::new(new_node))
    }

    fn insert(&mut self, key: K, value: V) -> Option<(K, Box<TreeNode<K, V>>)> {
        let mut subtree = self.choose_subtree(key.clone());
        match subtree.insert(key, value){
            Some(res) => {
                let child_pivot = res.0;
                let child_node = res.1;
                self.insert_entry(child_pivot, child_node);
                if self.keys.len() > MAX_KEYS {
                    return Some(self.split())
                }
                None
            }
            None => {
                None
            }
        }
    }
}





#[test]
fn test_build_btree(){
    let mut btr = BTree::new();
    btr.root.insert(3, "BANANA");
    btr.root.insert(14, "peach");
    btr.root.insert(1, "apple");
    btr.root.insert(5,"fig");
    println!("{:?}",btr.root.get(3));

}

#[test]
fn test_build_bplustree(){
    let mut btr = BTree::new();
    let mut build = btr.insert(3, "BANANA").insert(14, "peach").insert(1, "apple").insert(5,"fig");
    println!("{:?}",build.root.get(14));

}