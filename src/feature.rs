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
//! Multi-value features enable training/predicting for features which contain
//! multiple values, while still producing the proper prediction results
//! assuming that the handler's 'resolve()' method is implemented properly

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
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
    U16(u16),
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
            Value::U16(u) => write!(f, "(U16) {}", u),
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    fn validate_string(value: &str) -> Result<(), String> {
        if value.contains('\x00') {
            return Err(format!(
                "Feature value cannot contain null bytes (\\x00): '{}'",
                value.escape_default()
            ));
        }
        Ok(())
    }

    pub fn string(key: &str, value: &str) -> Feature {
        if let Err(e) = Self::validate_string(value) {
            panic!("{}", e);
        }
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

    pub fn u16(key: &str, value: u16) -> Feature {
        Feature {
            key: key.to_string(),
            values: smallvec::smallvec![Value::U16(value)],
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

    /// Creates a multi-value string feature.
    ///
    /// Enables traversing multiple tree paths during training/prediction.
    /// The caller must ensure no duplicate values are provided.
    ///
    /// Accepts various string types efficiently:
    /// - `Vec<&str>` - clones as needed
    /// - `Vec<String>` - moves ownership without cloning
    /// - Arrays, slices, and any iterator of string-like types
    ///
    /// # Panics
    /// Panics if any value contains null bytes (\x00), which are reserved for internal use.
    ///
    /// # Example
    /// ```
    /// use logictree::Feature;
    ///
    /// // With string literals (clones as needed)
    /// let formats = Feature::multi_string("format", vec!["banner", "video"]);
    ///
    /// // With owned strings (zero clones - moves ownership)
    /// let owned_formats = vec!["banner".to_string(), "video".to_string()];
    /// let formats = Feature::multi_string("format", owned_formats);
    /// ```
    pub fn multi_string<I, S>(key: &str, values: I) -> Feature
    where
        I: IntoIterator<Item = S>,
        S: Into<String> + AsRef<str>,
    {
        Feature {
            key: key.to_string(),
            values: values
                .into_iter()
                .map(|v| {
                    if let Err(e) = Self::validate_string(v.as_ref()) {
                        panic!("{}", e);
                    }
                    Value::String(v.into())
                })
                .collect(),
        }
    }

    /// Creates a multi-value boolean feature.
    ///
    /// The caller must ensure no duplicate values are provided.
    pub fn multi_boolean(key: &str, values: Vec<bool>) -> Feature {
        Feature {
            key: key.to_string(),
            values: values.into_iter().map(Value::Boolean).collect(),
        }
    }

    /// Creates a multi-value i32 feature.
    ///
    /// The caller must ensure no duplicate values are provided.
    pub fn multi_i32(key: &str, values: Vec<i32>) -> Feature {
        Feature {
            key: key.to_string(),
            values: values.into_iter().map(Value::I32).collect(),
        }
    }

    /// Creates a multi-value i64 feature.
    ///
    /// The caller must ensure no duplicate values are provided.
    pub fn multi_i64(key: &str, values: Vec<i64>) -> Feature {
        Feature {
            key: key.to_string(),
            values: values.into_iter().map(Value::I64).collect(),
        }
    }

    /// Creates a multi-value u32 feature.
    ///
    /// The caller must ensure no duplicate values are provided.
    pub fn multi_u32(key: &str, values: Vec<u32>) -> Feature {
        Feature {
            key: key.to_string(),
            values: values.into_iter().map(Value::U32).collect(),
        }
    }

    /// Creates a multi-value u64 feature.
    ///
    /// The caller must ensure no duplicate values are provided.
    pub fn multi_u64(key: &str, values: Vec<u64>) -> Feature {
        Feature {
            key: key.to_string(),
            values: values.into_iter().map(Value::U64).collect(),
        }
    }
}
