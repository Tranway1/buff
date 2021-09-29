use std::{fmt, ops::Range};

use super::store::{self, Store};
use super::util;

const ARRAY_LIMIT: u64 = 4096;
const BITMAP_LENGTH: usize = 1024;

#[derive(PartialEq, Clone)]
pub struct Container {
    pub key: u16,
    pub len: u64,
    pub store: Store,
}

pub struct Iter<'a> {
    pub key: u16,
    inner: store::Iter<'a>,
}

impl Container {
    pub fn new(key: u16) -> Container {
        Container {
            key,
            len: 0,
            store: Store::Array(Vec::new()),
        }
    }

    pub fn new_bitmap(key: u16) -> Container {
        let mut bits = Box::new([0; BITMAP_LENGTH]);
        Container {
            key,
            len: 0,
            store: Store::Bitmap(bits),
        }
    }
}

impl Container {
    pub fn insert(&mut self, index: u16) -> bool {
        if self.store.insert(index) {
            self.len += 1;
            self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    pub fn insert_bitmap(&mut self, index: u16, word:u32) -> bool {
        if self.store.insert_bitword(index, word as u64) {
            self.len += word.count_ones() as u64;
            // self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    pub fn insert_range(&mut self, range: Range<u16>) -> u64 {
        // If the range is larger than the array limit, skip populating the
        // array to then have to convert it to a bitmap anyway.
        if matches!(self.store, Store::Array(_)) && range.end - range.start > ARRAY_LIMIT as u16 {
            self.store = self.store.to_bitmap()
        }

        let inserted = self.store.insert_range(range);
        self.len += inserted;
        self.ensure_correct_store();
        inserted
    }

    pub fn push(&mut self, index: u16) {
        if self.store.push(index) {
            self.len += 1;
            self.ensure_correct_store();
        }
    }

    pub fn remove(&mut self, index: u16) -> bool {
        if self.store.remove(index) {
            self.len -= 1;
            self.ensure_correct_store();
            true
        } else {
            false
        }
    }

    pub fn remove_range(&mut self, start: u32, end: u32) -> u64 {
        debug_assert!(start <= end);
        if start == end {
            return 0;
        }
        let result = self.store.remove_range(start, end);
        self.len -= result;
        self.ensure_correct_store();
        result
    }

    pub fn contains(&self, index: u16) -> bool {
        self.store.contains(index)
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.store.is_disjoint(&other.store)
    }

    pub fn is_subset(&self, other: &Self) -> bool {
        self.len <= other.len && self.store.is_subset(&other.store)
    }

    pub fn union_with(&mut self, other: &Self) {
        self.store.union_with(&other.store);
        self.len = self.store.len();
        self.ensure_correct_store();
    }

    pub fn intersect_with(&mut self, other: &Self) {
        self.store.intersect_with(&other.store);
        self.len = self.store.len();
        self.ensure_correct_store();
    }

    pub fn difference_with(&mut self, other: &Self) {
        self.store.difference_with(&other.store);
        self.len = self.store.len();
        self.ensure_correct_store();
    }

    pub fn symmetric_difference_with(&mut self, other: &Self) {
        self.store.symmetric_difference_with(&other.store);
        self.len = self.store.len();
        self.ensure_correct_store();
    }

    pub fn min(&self) -> u16 {
        self.store.min()
    }

    pub fn max(&self) -> u16 {
        self.store.max()
    }

    pub fn ensure_correct_store(&mut self) {
        let new_store = match (&self.store, self.len) {
            (store @ &Store::Bitmap(..), len) if len <= ARRAY_LIMIT => Some(store.to_array()),
            (store @ &Store::Array(..), len) if len > ARRAY_LIMIT => Some(store.to_bitmap()),
            _ => None,
        };
        if let Some(new_store) = new_store {
            self.store = new_store;
        }
    }
}

impl<'a> IntoIterator for &'a Container {
    type Item = u32;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        Iter {
            key: self.key,
            inner: (&self.store).into_iter(),
        }
    }
}

impl IntoIterator for Container {
    type Item = u32;
    type IntoIter = Iter<'static>;

    fn into_iter(self) -> Iter<'static> {
        Iter {
            key: self.key,
            inner: self.store.into_iter(),
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.inner.next().map(|i| util::join(self.key, i))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        panic!("Should never be called (roaring::Iter caches the size_hint itself)")
    }
}

impl fmt::Debug for Container {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        format!("Container<{:?} @ {:?}>", self.len, self.key).fmt(formatter)
    }
}
