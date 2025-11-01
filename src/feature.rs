//! Feature input types for single-value and multi-value features.
//!
//! This module provides the [`Feature`] and [`Value`] types used to represent
//! input data to the tree. Features can have single or multiple values.
//!
//! # Single vs Multi-Value Features
//!
//! - **Single-value**: Use `Feature::string()`, `Feature::i32()`, etc.
//! - **Multi-value**: Use `Feature::multi_string()`, `Feature::multi_i32()`, etc.
//!
//! Multi-value features enable training/predicting across multiple tree paths
//! simultaneously, crucial for scenarios like ad formats, content categories, etc.
//!
//! # Performance
//!
//! Single-value features use `SmallVec` for zero-allocation inline storage.
//! Multi-value features automatically deduplicate values using `HashSet`.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashSet;
use std::fmt;

/// Define an input feature value and its type.
///
/// Supports string, boolean, and integer types. All values must implement
/// Hash, Eq, and PartialEq for use as tree node keys.
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

/// A feature input defined by its key name and value(s).
///
/// Supports both single-value and multi-value features:
/// - **Single-value**: Created via `Feature::string()`, `Feature::i32()`, etc.
/// - **Multi-value**: Created via `Feature::multi_string()`, `Feature::multi_i32()`, etc.
///
/// # Multi-Value Features
///
/// Multi-value features enable traversing multiple tree paths simultaneously during
/// training and prediction. When training with `format=[banner, video]`:
/// - Parent nodes (country, device) are visited once
/// - Child nodes (banner AND video) are both created/updated
/// - Each child receives the full training input (no fractional attribution)
///
/// This prevents double-counting in parent nodes while maintaining rich child statistics.
///
/// # Performance
///
/// Uses `SmallVec<[Value; 1]>` for inline storage:
/// - **Single values**: Zero heap allocations (stored inline)
/// - **Multiple values**: Heap allocation only when needed
///
/// # Deduplication
///
/// Multi-value constructors automatically deduplicate values using `HashSet`.
/// `Feature::multi_string("format", vec!["banner", "banner", "video"])` stores `["banner", "video"]`.
///
/// # Examples
///
/// ```
/// use logictree::Feature;
///
/// // Single-value feature (zero allocation)
/// let country = Feature::string("country", "USA");
///
/// // Multi-value feature (automatic deduplication)
/// let formats = Feature::multi_string("format", vec!["banner", "video", "banner"]);
/// // Internally stores: ["banner", "video"]
/// ```
#[derive(Clone, Debug)]
pub struct Feature {
    /// Feature name (must match tree structure)
    pub key: String,
    /// One or more values for this feature
    pub values: SmallVec<[Value; 1]>,
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.values.len() == 1 {
            write!(f, "{} -> {}", self.key, self.values[0])
        } else {
            write!(f, "{} -> [", self.key)?;
            for (i, val) in self.values.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
            }
            write!(f, "]")
        }
    }
}

impl Feature {
    pub fn string(key: &str, value: &str) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::String(value.to_string())],
        }
    }

    pub fn boolean(key: &str, value: bool) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::Boolean(value)],
        }
    }

    pub fn i32(key: &str, value: i32) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::I32(value)],
        }
    }

    pub fn i64(key: &str, value: i64) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::I64(value)],
        }
    }

    pub fn u32(key: &str, value: u32) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::U32(value)],
        }
    }

    pub fn u64(key: &str, value: u64) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::U64(value)],
        }
    }

    /// Creates a multi-value string feature with automatic deduplication.
    ///
    /// Enables traversing multiple tree paths during training/prediction.
    /// Duplicates are automatically removed.
    ///
    /// # Example
    /// ```
    /// use logictree::Feature;
    ///
    /// // For OpenRTB bid request supporting multiple ad formats
    /// let formats = Feature::multi_string("format", vec!["banner", "video"]);
    /// ```
    pub fn multi_string(key: &str, values: Vec<&str>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(|v| Value::String(v.to_string())).collect(),
        }
    }

    /// Creates a multi-value boolean feature with automatic deduplication.
    pub fn multi_boolean(key: &str, values: Vec<bool>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(Value::Boolean).collect(),
        }
    }

    /// Creates a multi-value i32 feature with automatic deduplication.
    pub fn multi_i32(key: &str, values: Vec<i32>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(Value::I32).collect(),
        }
    }

    /// Creates a multi-value i64 feature with automatic deduplication.
    pub fn multi_i64(key: &str, values: Vec<i64>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(Value::I64).collect(),
        }
    }

    /// Creates a multi-value u32 feature with automatic deduplication.
    pub fn multi_u32(key: &str, values: Vec<u32>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(Value::U32).collect(),
        }
    }

    /// Creates a multi-value u64 feature with automatic deduplication.
    pub fn multi_u64(key: &str, values: Vec<u64>) -> Feature {
        let unique: HashSet<_> = values.into_iter().collect();
        Feature {
            key: key.to_string(),
            values: unique.into_iter().map(Value::U64).collect(),
        }
    }
}
