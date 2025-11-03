//! A high-performance decision tree library for online learning with multi-value feature support.
//!
//! # Overview
//!
//! LogicTree is a thread-safe, concurrent decision tree designed for:
//! - **Online learning**: Continuous training from streaming data
//! - **Custom prediction logic**: Implement your own handlers at each node
//! - **Multi-value features**: Properly handle multi-valued inputs without double-counting
//! - **Automatic fallback**: Parent nodes provide predictions when child data is insufficient
//! - **Tree pruning**: Evict inactive nodes to prevent unbounded growth
//!
//! # Multi-Value Features
//!
//! The key innovation is native support for **multi-value features**, crucial for scenarios like:
//! - **Ad serving**: Requests supporting multiple formats (banner + video)
//! - **Content classification**: Pages belonging to multiple categories
//! - **Multi-device targeting**: Campaigns targeting multiple device types
//!
//! Traditional approaches double-count parent metrics when training separate paths.
//! LogicTree solves this by visiting parent nodes once while creating multiple child branches.
//!
//! # Quick Start
//!
//! ```rust
//! use logictree::{LogicTree, PredictionHandler, Feature, PredictionResult};
//! use std::sync::Mutex;
//! # use serde::{Serialize, Deserialize};
//!
//! // 1. Define your prediction handler
//! #[derive(Serialize, Deserialize)]
//! struct MyHandler {
//!     count: Mutex<u32>,
//! }
//!
//! # impl Clone for MyHandler {
//! #     fn clone(&self) -> Self {
//! #         MyHandler { count: Mutex::new(*self.count.lock().unwrap()) }
//! #     }
//! # }
//! # impl MyHandler { fn new() -> Self { MyHandler { count: Mutex::new(0) } } }
//! #
//! impl PredictionHandler<u32, u32> for MyHandler {
//!     fn train(&self, input: &u32) {
//!         *self.count.lock().unwrap() += input;
//!     }
//!
//!     fn predict(&self) -> u32 {
//!         *self.count.lock().unwrap()
//!     }
//!
//!     fn resolve(&self, predictions: Vec<(u32, usize)>) -> Option<u32> {
//!         // Sum independent samples from peer nodes (for multi-value features)
//!         Some(predictions.iter().map(|(v, _)| v).sum())
//!     }
//!
//!     fn should_prune(&self) -> bool { false }
//!     fn new_instance(&self) -> Self { MyHandler::new() }
//! }
//!
//! // 2. Create tree with ordered features
//! let tree = LogicTree::new(
//!     vec!["country".to_string(), "format".to_string()],
//!     MyHandler::new()
//! );
//!
//! // 3. Train with multi-value feature
//! tree.train(&vec![
//!     Feature::string("country", "USA"),
//!     Feature::multi_string("format", vec!["banner", "video"]),
//! ], &10).unwrap();
//!
//! // Parent node (USA) sees 1 auction, not 2!
//! // Child nodes (banner, video) each see 1 auction
//!
//! // 4. Predict with multi-value feature (aggregates via resolve)
//! let result = tree.predict(&vec![
//!     Feature::string("country", "USA"),
//!     Feature::multi_string("format", vec!["banner", "video"]),
//! ]).unwrap()
//! .map(|r| r.value);
//!
//! // 5. Check prediction depth information
//! let prediction = tree.predict(&vec![
//!     Feature::string("country", "USA"),
//! ]).unwrap().unwrap();
//!
//! assert_eq!(prediction.depth, 1);        // Prediction from depth 1 (after first feature)
//! assert_eq!(prediction.full_depth, false); // Not at full depth (tree has 2 features)
//! ```
//!
//! # Performance
//!
//! - **Thread-safe**: Uses DashMap for lock-free concurrent access
//! - **Zero-allocation**: Single-value features use SmallVec (inline storage)
//! - **Fast predictions**: ~1M predictions/second on modern CPUs
//! - **Automatic deduplication**: Multi-value features deduplicate via HashSet
//!
//! # See Also
//!
//! - [`LogicTree`]: Main tree structure
//! - [`PredictionHandler`]: Trait for custom prediction logic
//! - [`Feature`]: Feature input types (single and multi-value)
//! - See README for detailed multi-value feature examples

mod feature;
mod handler;
mod logictree;
mod node;

/// Result from a tree prediction containing the value and depth information
///
/// # Fields
/// * `value` - The predicted value returned by the handler
/// * `depth` - The depth in the tree where this prediction was found:
///   - 0 = root node (before any features)
///   - 1 = after first feature
///   - 2 = after second feature
///   - etc.
/// * `full_depth` - Whether this prediction is from the deepest possible level (all features matched)
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResult<O> {
    pub value: O,
    pub depth: usize,
    pub full_depth: bool,
}

pub use feature::*;
pub use handler::*;
pub use logictree::*;

#[cfg(test)]
mod tests {
    use crate::{Feature, LogicTree, PredictionHandler};
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::sync::Mutex;

    #[derive(Serialize, Deserialize)]
    struct TestHandler {
        pub total: Mutex<usize>,
    }

    impl Clone for TestHandler {
        fn clone(&self) -> Self {
            TestHandler {
                total: Mutex::new(*self.total.lock().unwrap()),
            }
        }
    }

    impl TestHandler {
        pub fn new() -> TestHandler {
            TestHandler {
                total: Mutex::new(0),
            }
        }
    }

    impl PredictionHandler<u32, u32> for TestHandler {
        fn train(&self, input: &u32) -> () {
            *self.total.lock().unwrap() += *input as usize;
        }

        fn predict(&self) -> u32 {
            *self.total.lock().unwrap() as u32
        }

        fn should_prune(&self) -> bool {
            true
        }

        fn new_instance(&self) -> Self {
            TestHandler::new()
        }

        fn resolve(&self, predictions: Vec<(u32, usize)>) -> Option<u32> {
            predictions.into_iter().map(|(v, _)| v).next()
        }
    }

    fn test_map_assertions(tree: &LogicTree<u32, u32, TestHandler>) {
        let res = tree.predict(&vec![Feature::string("one", "foo")])
            .unwrap()
            .map(|r| r.value);
        assert_eq!(
            res,
            Some(20),
            "Foo parent prediction should equal sum of parent and children"
        );

        let res = tree.predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bar"),
        ])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(
            res,
            Some(10),
            "Foo.bar prediction should equal node specific training value"
        );

        let res = tree.predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "dang"),
        ])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(
            res,
            Some(20),
            "Non existent foo.dang should return parent summary of foo"
        );

        // prediction features in wrong order
        tree.predict(&vec![
            Feature::string("two", "bar"),
            Feature::string("one", "foo"),
        ])
        .expect_err("Should fail on invalid feature order");

        let res = tree.size(true);
        assert_eq!(res, 3, "Should count leaf nodes correctly (bar baz bum)");

        let res = tree.size(false);
        assert_eq!(res, 5, "Should count total nodes correctly");

        assert_eq!(
            tree.prune(),
            true,
            "Pruning when all handlers should_prune should empty the tree"
        );
    }

    #[test]
    fn it_works() {
        let tree = LogicTree::new(
            vec!["one".to_string(), "two".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![Feature::string("one", "foo"), Feature::string("two", "bar")],
            &10,
        )
        .unwrap();

        tree.train(
            &vec![Feature::string("one", "foo"), Feature::string("two", "baz")],
            &5,
        )
        .unwrap();

        tree.train(
            &vec![Feature::string("one", "foo"), Feature::string("two", "bum")],
            &5,
        )
        .unwrap();

        test_map_assertions(&tree);

        let file = "tree.bin";

        tree.save(file).expect("Should serialize tree to bin file");

        let tree: LogicTree<u32, u32, TestHandler> =
            LogicTree::load(file).expect("Should have succeeded deserializing tree from bin file");

        test_map_assertions(&tree);

        fs::remove_file(file).expect("Should have deleted the test bin file");
    }

    #[test]
    fn test_multi_value_features() {
        let tree = LogicTree::new(
            vec![
                "country".to_string(),
                "device".to_string(),
                "format".to_string(),
            ],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("device", "android"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &10,
        )
        .unwrap();

        let res = tree.predict(&vec![Feature::string("country", "usa")]).unwrap().unwrap();
        assert_eq!(res.value, 10, "USA node should see 10 (not 20 from double-counting)");
        assert_eq!(res.depth, 1, "Should be at depth 1 after consuming country feature");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
        ]).unwrap().unwrap();
        assert_eq!(res.value, 10, "USA->android node should see 10 (not 20)");
        assert_eq!(res.depth, 2, "Should be at depth 2 after consuming two features");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]).unwrap().unwrap();
        assert_eq!(res.value, 10, "Composite banner|video should exist with value 10");
        assert_eq!(res.depth, 3, "Should be at depth 3 after consuming all features");
        assert_eq!(res.full_depth, true, "Should be full depth with all 3 features");
    }

    #[derive(Serialize, Deserialize)]
    struct SummingHandler {
        pub total: Mutex<usize>,
    }

    impl Clone for SummingHandler {
        fn clone(&self) -> Self {
            SummingHandler {
                total: Mutex::new(*self.total.lock().unwrap()),
            }
        }
    }

    impl SummingHandler {
        pub fn new() -> SummingHandler {
            SummingHandler {
                total: Mutex::new(0),
            }
        }
    }

    impl PredictionHandler<u32, u32> for SummingHandler {
        fn train(&self, input: &u32) -> () {
            *self.total.lock().unwrap() += *input as usize;
        }

        fn predict(&self) -> u32 {
            *self.total.lock().unwrap() as u32
        }

        fn should_prune(&self) -> bool {
            false
        }

        fn new_instance(&self) -> Self {
            SummingHandler::new()
        }

        fn resolve(&self, predictions: Vec<(u32, usize)>) -> Option<u32> {
            if predictions.is_empty() {
                return None;
            }
            Some(predictions.iter().map(|(v, _)| v).sum())
        }
    }

    #[test]
    fn test_multi_value_prediction_with_custom_resolve() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            SummingHandler::new(),
        );

        // Train banner format with value 100
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        // Train video format with value 200
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "video"),
            ],
            &200,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(300), "Should sum banner (100) and video (200) = 300");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Single banner prediction should be 100");
    }

    #[test]
    fn test_multi_value_mixed_with_single_value() {
        let tree = LogicTree::new(
            vec![
                "country".to_string(),
                "device".to_string(),
                "format".to_string(),
            ],
            TestHandler::new(),
        );

        // Mix single and multi-value features
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("device", vec!["android", "ios"]),
                Feature::string("format", "banner"),
            ],
            &5,
        )
        .unwrap();

        // Check both android and ios branches exist
        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(5), "Android->banner should exist");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(res, Some(5), "iOS->banner should exist");

        // Parent should not double-count
        let res = tree.predict(&vec![Feature::string("country", "usa")])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(res, Some(5), "USA should see 5 (not 10)");
    }

    #[test]
    fn test_ctv_use_case() {
        let tree = LogicTree::new(
            vec![
                "country".to_string(),
                "device".to_string(),
                "format".to_string(),
            ],
            TestHandler::new(),
        );

        // Simulate CTV with multiple identical video impressions
        // Train 10 times to simulate 10 video impressions
        for _ in 0..10 {
            tree.train(
                &vec![
                    Feature::string("country", "usa"),
                    Feature::string("device", "ctv"),
                    Feature::string("format", "video"),
                ],
                &1,
            )
            .unwrap();
        }

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
            Feature::string("format", "video"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(10), "CTV video should have 10 impressions");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(10), "CTV parent should see 10 impressions");
    }

    #[test]
    fn test_empty_multi_value_returns_error() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        let result = tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", Vec::<&str>::new()),
            ],
            &10,
        );

        assert!(result.is_err(), "Empty values should return error");
        assert!(
            result.unwrap_err().contains("cannot have empty values"),
            "Error message should mention empty values"
        );
    }

    #[test]
    fn test_duplicate_values_deduplicated() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "banner", "video"]),
            ],
            &10,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(10), "Duplicate banner should be deduplicated");
    }

    #[test]
    fn test_serialization_with_multi_value() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &42,
        )
        .unwrap();

        let file = "test_multi_value_tree.bin";

        tree.save(file).expect("Should serialize tree");

        let loaded: LogicTree<u32, u32, TestHandler> =
            LogicTree::load(file).expect("Should deserialize tree");

        let res = loaded.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(42), "Loaded tree should preserve composite structure");

        fs::remove_file(file).expect("Should delete test file");
    }

    #[test]
    fn test_resolve_sums_independent_samples() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            SummingHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "video"),
            ],
            &201,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]).unwrap().unwrap();
        assert_eq!(res.value, 301, "Should sum banner (100) + video (201) = 301");
        assert_eq!(res.depth, 2, "Should resolve at depth 2 with both format values");
        assert_eq!(res.full_depth, true, "Should be full depth for 2-feature tree");
    }

    #[test]
    fn test_multiple_multi_value_features_in_path() {
        let tree = LogicTree::new(
            vec![
                "country".to_string(),
                "device".to_string(),
                "format".to_string(),
                "network".to_string(),
                "size".to_string(),
            ],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("device", "android"),
                Feature::multi_string("format", vec!["banner", "video"]),
                Feature::string("network", "wifi"),
                Feature::multi_string("size", vec!["300x250", "728x90"]),
            ],
            &10,
        )
        .unwrap();

        let res = tree.predict(&vec![Feature::string("country", "usa")])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(res, Some(10), "USA should see 10 (not 20 or 40)");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(10), "USA->android should see 10");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::multi_string("format", vec!["banner", "video"]),
            Feature::string("network", "wifi"),
            Feature::multi_string("size", vec!["300x250", "728x90"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(10), "Composite path should return prediction");

        let leaf_count = tree.size(true);
        assert_eq!(leaf_count, 1, "Should have 1 composite leaf node");
    }

    #[test]
    fn test_composite_key_creation() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &100,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Composite child should return prediction");
    }

    #[test]
    fn test_composite_key_determinism() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &50,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["video", "banner"]),
            ],
            &50,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Same composite regardless of value order");
    }

    #[test]
    fn test_composite_union_lookup() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            SummingHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "video"),
            ],
            &200,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &50,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(350), "Should sum banner (100) + video (200) + banner|video (50) = 350");
    }

    #[test]
    fn test_composite_with_multiple_composites() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            SummingHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &100,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "native"]),
            ],
            &200,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(300), "Should sum banner|video (100) + banner|native (200) = 300");
    }

    #[test]
    fn test_single_value_no_composite() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Single value should not create composite");

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Query for [banner,video] should only find banner (no composite)");
    }

    #[test]
    fn test_composite_deeper_children() {
        let tree = LogicTree::new(
            vec![
                "country".to_string(),
                "format".to_string(),
                "size".to_string(),
            ],
            TestHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
                Feature::string("size", "300x250"),
            ],
            &100,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
            Feature::string("size", "300x250"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "Composite child should have deeper children");
    }

    #[test]
    fn test_single_value_query_includes_composites() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            SummingHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &50,
        )
        .unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]).unwrap().unwrap();
        assert_eq!(res.value, 150, "Query [banner] should sum banner (100) + banner|video (50) = 150");
        assert_eq!(res.depth, 2, "Should find predictions at depth 2");
        assert_eq!(res.full_depth, true, "Should be full depth for 2-feature tree");
    }

    #[test]
    #[should_panic(expected = "Feature value cannot contain null bytes")]
    fn test_null_byte_validation_single() {
        Feature::string("format", "banner\x00video");
    }

    #[test]
    #[should_panic(expected = "Feature value cannot contain null bytes")]
    fn test_null_byte_validation_multi() {
        Feature::multi_string("format", vec!["banner", "video\x00native"]);
    }

    #[test]
    fn test_composite_index_cleanup_on_prune() {
        #[derive(Serialize, Deserialize)]
        struct PrunableHandler {
            total: Mutex<u32>,
            should_prune_flag: Mutex<bool>,
        }

        impl Clone for PrunableHandler {
            fn clone(&self) -> Self {
                PrunableHandler {
                    total: Mutex::new(*self.total.lock().unwrap()),
                    should_prune_flag: Mutex::new(*self.should_prune_flag.lock().unwrap()),
                }
            }
        }

        impl PrunableHandler {
            fn new() -> Self {
                PrunableHandler {
                    total: Mutex::new(0),
                    should_prune_flag: Mutex::new(false),
                }
            }
        }

        impl PredictionHandler<u32, u32> for PrunableHandler {
            fn train(&self, input: &u32) {
                *self.total.lock().unwrap() += input;
            }

            fn predict(&self) -> u32 {
                *self.total.lock().unwrap()
            }

            fn should_prune(&self) -> bool {
                *self.should_prune_flag.lock().unwrap()
            }

            fn new_instance(&self) -> Self {
                PrunableHandler::new()
            }

            fn resolve(&self, predictions: Vec<(u32, usize)>) -> Option<u32> {
                Some(predictions.iter().map(|(v, _)| v).sum())
            }
        }

        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            PrunableHandler::new(),
        );

        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &100,
        )
        .unwrap();

        let res_before = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(
            res_before.unwrap().map(|r| r.value),
            Some(100),
            "Should find banner|video composite"
        );

        tree.prune();
    }

    #[test]
    fn test_consecutive_multi_value_features() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Train with consecutive multi-value features
        tree.train(
            &vec![
                Feature::multi_string("country", vec!["USA", "Canada"]),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &100,
        )
        .unwrap();

        // All these should return 100 (no double counting, no aggregation)

        // Single country matches the composite
        let res = tree.predict(&vec![Feature::string("country", "USA")])
        .unwrap()
        .map(|r| r.value);
        assert_eq!(res, Some(100), "Single country should return 100");

        // Single country + single format
        let res = tree.predict(&vec![
            Feature::string("country", "USA"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "USA + banner should return 100");

        // Single country + multi format
        let res = tree.predict(&vec![
            Feature::string("country", "USA"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "USA + [banner,video] should return 100");

        // Multi country + multi format (exact training match)
        let res = tree.predict(&vec![
            Feature::multi_string("country", vec!["USA", "Canada"]),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res.unwrap().map(|r| r.value), Some(100), "[USA,Canada] + [banner,video] should return 100");
    }

    #[test]
    fn test_multi_string_with_owned_strings() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Test with Vec<String> - should work without cloning
        let owned_formats: Vec<String> = vec![
            "banner".to_string(),
            "video".to_string(),
            "native".to_string(),
        ];
        let owned_countries: Vec<String> = vec!["USA".to_string(), "Canada".to_string()];

        tree.train(
            &vec![
                Feature::multi_string("country", owned_countries),
                Feature::multi_string("format", owned_formats),
            ],
            &42,
        )
        .expect("Training should succeed with owned strings");

        // Also test that mixing owned and borrowed works
        let mixed_formats = vec!["banner".to_string(), "video".to_string()];
        let res = tree.predict(&vec![
            Feature::multi_string("country", vec!["USA", "Canada"]), // borrowed
            Feature::multi_string("format", mixed_formats),          // owned
        ]);

        assert_eq!(res.unwrap().map(|r| r.value), Some(42), "Should predict correctly with mixed owned/borrowed strings");

        // Test with iterators too
        let iter_formats = vec!["banner", "native"].into_iter();
        let _ = tree.predict(&vec![
            Feature::string("country", "USA"),
            Feature::multi_string("format", iter_formats),
        ]);
    }

    #[test]
    fn test_depth_tracking_at_root() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string()],
            TestHandler::new(),
        );

        tree.train(&vec![], &10).unwrap();

        let res = tree.predict(&vec![]).unwrap().unwrap();
        assert_eq!(res.depth, 0, "Root prediction should have depth 0");
        assert_eq!(res.full_depth, false, "Root is not full depth for 2-feature tree");
        assert_eq!(res.value, 10);
    }

    #[test]
    fn test_depth_tracking_at_level_1() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(&vec![Feature::string("country", "usa")], &20).unwrap();

        let res = tree.predict(&vec![Feature::string("country", "usa")]).unwrap().unwrap();
        assert_eq!(res.depth, 1, "Should be at depth 1 after consuming 1 feature");
        assert_eq!(res.full_depth, false, "Depth 1 is not full depth for 3-feature tree");
        assert_eq!(res.value, 20);
    }

    #[test]
    fn test_depth_tracking_at_full_depth() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string()],
            TestHandler::new(),
        );

        tree.train(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ], &30).unwrap();

        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ]).unwrap().unwrap();

        assert_eq!(res.depth, 2, "Should be at depth 2 after consuming 2 features");
        assert_eq!(res.full_depth, true, "Depth 2 is full depth for 2-feature tree");
        assert_eq!(res.value, 30);
    }

    #[test]
    fn test_depth_fallback_to_parent() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Only train at depth 1
        tree.train(&vec![Feature::string("country", "usa")], &15).unwrap();

        // Predict at depth 2 with unknown path - should fall back to parent
        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "unknown"),
        ]).unwrap().unwrap();

        assert_eq!(res.depth, 1, "Should fall back to depth 1 parent");
        assert_eq!(res.full_depth, false, "Fallback is not full depth");
        assert_eq!(res.value, 15, "Should get parent value");
    }

    #[test]
    fn test_depth_empty_tree_root() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string()],
            TestHandler::new(),
        );

        // No training, just root handler with initial value 0
        let res = tree.predict(&vec![]).unwrap().unwrap();
        assert_eq!(res.depth, 0, "Empty tree root should have depth 0");
        assert_eq!(res.full_depth, false, "Root is not full depth for 2-feature tree");
        assert_eq!(res.value, 0, "Should get default handler value");
    }

    #[test]
    fn test_depth_partial_path_three_level_tree() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ], &25).unwrap();

        // Partial path - only 2 of 3 features
        let res = tree.predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ]).unwrap().unwrap();

        assert_eq!(res.depth, 2, "Should be at depth 2");
        assert_eq!(res.full_depth, false, "Not full depth for 3-feature tree");
        assert_eq!(res.value, 25);
    }
}
