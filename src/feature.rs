use std::fmt;
use serde::{Deserialize, Serialize};

/// Define an input feature value and its type
#[derive(Hash, Eq, PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum Value {
    String(String),
    Boolean(bool),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "(String) {}", s),
            Value::Boolean(b) => write!(f, "(Boolean) {}", b),
            Value::I32(i) => write!(f, "(I32) {}", i),
            Value::I64(i) => write!(f, "(I64) {}", i),
            Value::U32(u) => write!(f, "(U32) {}", u),
            Value::U64(u) => write!(f, "(U64) {}", u),
        }
    }
}

/// A feature input as defined by its key name and value
#[derive(Clone, Debug)]
pub struct Feature {
    pub key: String,
    pub value: Value,
}

impl<'a> fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} -> {}", self.key, self.value)
    }
}

impl Feature {
    pub fn string(key: &str, value: &'static str) -> Feature {
        Feature { key: key.to_string(), value: Value::String(value.to_string()) }
    }

    pub fn boolean(key: &str, value: bool) -> Feature {
        Feature {  key: key.to_string(), value: Value::Boolean(value) }
    }

    pub fn i32(key: &str, value: i32) -> Feature {
        Feature { key: key.to_string(), value: Value::I32(value) }
    }

    pub fn i64(key: &str, value: i64) -> Feature {
        Feature {  key: key.to_string(), value: Value::I64(value) }
    }

    pub fn u32(key: &str, value: u32) -> Feature {
        Feature {  key: key.to_string(), value: Value::U32(value) }
    }

    pub fn u64(key: &str, value: u64) -> Feature {
        Feature {  key: key.to_string(), value: Value::U64(value) }
    }

}