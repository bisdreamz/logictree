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
//! use logictree::{LogicTree, PredictionHandler, Feature};
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
//!     fn predict(&self) -> Option<u32> {
//!         Some(*self.count.lock().unwrap())
//!     }
//!
//!     fn fold(&self, predictions: Vec<u32>) -> Option<u32> {
//!         // Average predictions from peer nodes (for multi-value features)
//!         Some(predictions.iter().sum::<u32>() / predictions.len() as u32)
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
//! tree.train(vec![
//!     Feature::string("country", "USA"),
//!     Feature::multi_string("format", vec!["banner", "video"]),
//! ], &10).unwrap();
//!
//! // Parent node (USA) sees 1 auction, not 2!
//! // Child nodes (banner, video) each see 1 auction
//!
//! // 4. Predict with multi-value feature (aggregates via fold)
//! let result = tree.predict(vec![
//!     Feature::string("country", "USA"),
//!     Feature::multi_string("format", vec!["banner", "video"]),
//! ]).unwrap();
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
//! - [README Multi-Value Section](https://github.com/yourusername/logictree#multi-value-features-solving-the-attribution-problem)

mod feature;
mod handler;
mod logictree;
mod node;

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

    // TODO handler examples with complex i/o types
    impl PredictionHandler<u32, u32> for TestHandler {
        fn train(&self, input: &u32) -> () {
            *self.total.lock().unwrap() += *input as usize;
        }

        fn predict(&self) -> Option<u32> {
            Some(*self.total.lock().unwrap() as u32)
        }

        fn should_prune(&self) -> bool {
            true
        }

        fn new_instance(&self) -> Self {
            TestHandler::new()
        }

        fn fold(&self, predictions: Vec<u32>) -> Option<u32> {
            predictions.into_iter().next()
        }
    }

    fn test_map_assertions(tree: &LogicTree<u32, u32, TestHandler>) {
        let res = tree.predict(vec![Feature::string("one", "foo")]);
        assert_eq!(
            res,
            Ok(Some(20)),
            "Foo parent prediction should equal sum of parent and children"
        );

        let res = tree.predict(vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bar"),
        ]);
        assert_eq!(
            res,
            Ok(Some(10)),
            "Foo.bar prediction should equal node specific training value"
        );

        let res = tree.predict(vec![
            Feature::string("one", "foo"),
            Feature::string("two", "dang"),
        ]);
        assert_eq!(
            res,
            Ok(Some(20)),
            "Non existent foo.dang should return parent summary of foo"
        );

        // prediction features in wrong order
        tree.predict(vec![
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
            vec![Feature::string("one", "foo"), Feature::string("two", "bar")],
            &10,
        )
        .unwrap();

        tree.train(
            vec![Feature::string("one", "foo"), Feature::string("two", "baz")],
            &5,
        )
        .unwrap();

        tree.train(
            vec![Feature::string("one", "foo"), Feature::string("two", "bum")],
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
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Train with multi-value format feature: [banner, video]
        tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::string("device", "android"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &10,
        )
        .unwrap();

        // Verify parent nodes see correct counts (no double-counting)
        let res = tree.predict(vec![
            Feature::string("country", "usa"),
        ]);
        assert_eq!(res, Ok(Some(10)), "USA node should see 10 (not 20 from double-counting)");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
        ]);
        assert_eq!(res, Ok(Some(10)), "USA->android node should see 10 (not 20)");

        // Verify both banner and video branches were created and trained
        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(10)), "Banner branch should exist with value 10");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "video"),
        ]);
        assert_eq!(res, Ok(Some(10)), "Video branch should exist with value 10");
    }

    #[derive(Serialize, Deserialize)]
    struct AveragingHandler {
        pub total: Mutex<usize>,
    }

    impl Clone for AveragingHandler {
        fn clone(&self) -> Self {
            AveragingHandler {
                total: Mutex::new(*self.total.lock().unwrap()),
            }
        }
    }

    impl AveragingHandler {
        pub fn new() -> AveragingHandler {
            AveragingHandler {
                total: Mutex::new(0),
            }
        }
    }

    impl PredictionHandler<u32, u32> for AveragingHandler {
        fn train(&self, input: &u32) -> () {
            *self.total.lock().unwrap() += *input as usize;
        }

        fn predict(&self) -> Option<u32> {
            Some(*self.total.lock().unwrap() as u32)
        }

        fn should_prune(&self) -> bool {
            false
        }

        fn new_instance(&self) -> Self {
            AveragingHandler::new()
        }

        fn fold(&self, predictions: Vec<u32>) -> Option<u32> {
            if predictions.is_empty() {
                return None;
            }
            let sum = predictions.iter().sum::<u32>();
            let len = predictions.len() as u32;
            Some((sum + len / 2) / len)
        }
    }

    #[test]
    fn test_multi_value_prediction_with_custom_fold() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            AveragingHandler::new(),
        );

        // Train banner format with value 100
        tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::string("format", "banner"),
            ],
            &100,
        )
        .unwrap();

        // Train video format with value 200
        tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::string("format", "video"),
            ],
            &200,
        )
        .unwrap();

        // Predict with multi-value format - should average banner (100) and video (200) = 150
        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res, Ok(Some(150)), "Should average banner (100) and video (200) predictions");

        // Single-value prediction should still work
        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(100)), "Single banner prediction should be 100");
    }

    #[test]
    fn test_multi_value_mixed_with_single_value() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Mix single and multi-value features
        tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::multi_string("device", vec!["android", "ios"]),
                Feature::string("format", "banner"),
            ],
            &5,
        )
        .unwrap();

        // Check both android and ios branches exist
        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(5)), "Android->banner should exist");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(5)), "iOS->banner should exist");

        // Parent should not double-count
        let res = tree.predict(vec![Feature::string("country", "usa")]);
        assert_eq!(res, Ok(Some(5)), "USA should see 5 (not 10)");
    }

    #[test]
    fn test_ctv_use_case() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "device".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        // Simulate CTV with multiple identical video impressions
        // Train 10 times to simulate 10 video impressions
        for _ in 0..10 {
            tree.train(
                vec![
                    Feature::string("country", "usa"),
                    Feature::string("device", "ctv"),
                    Feature::string("format", "video"),
                ],
                &1,
            )
            .unwrap();
        }

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
            Feature::string("format", "video"),
        ]);
        assert_eq!(res, Ok(Some(10)), "CTV video should have 10 impressions");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
        ]);
        assert_eq!(res, Ok(Some(10)), "CTV parent should see 10 impressions");
    }

    #[test]
    fn test_empty_multi_value_returns_error() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        let result = tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec![]),
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
            vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "banner", "video"]),
            ],
            &10,
        )
        .unwrap();

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(10)), "Duplicate banner should be deduplicated");
    }

    #[test]
    fn test_serialization_with_multi_value() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            TestHandler::new(),
        );

        tree.train(
            vec![
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

        let res = loaded.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ]);
        assert_eq!(res, Ok(Some(42)), "Loaded tree should preserve multi-value structure");

        let res = loaded.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
        ]);
        assert_eq!(res, Ok(Some(42)), "Both branches should exist after deserialization");

        fs::remove_file(file).expect("Should delete test file");
    }

    #[test]
    fn test_integer_rounding_in_fold() {
        let tree = LogicTree::new(
            vec!["country".to_string(), "format".to_string()],
            AveragingHandler::new(),
        );

        tree.train(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ], &100).unwrap();

        tree.train(vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
        ], &201).unwrap();

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ]);
        assert_eq!(res, Ok(Some(151)), "Should round 150.5 up to 151");
    }

    #[test]
    fn test_multiple_multi_value_features_in_path() {
        // Real-world scenario: [country, device, format, network, size]
        // with multi-value at format and size positions
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

        // Train with multiple multi-value features in same path
        tree.train(
            vec![
                Feature::string("country", "usa"),
                Feature::string("device", "android"),
                Feature::multi_string("format", vec!["banner", "video"]),
                Feature::string("network", "wifi"),
                Feature::multi_string("size", vec!["300x250", "728x90"]),
            ],
            &10,
        )
        .unwrap();

        // Verify parent nodes see correct counts (no multiplication)
        let res = tree.predict(vec![Feature::string("country", "usa")]);
        assert_eq!(res, Ok(Some(10)), "USA should see 10 (not 20 or 40)");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
        ]);
        assert_eq!(res, Ok(Some(10)), "USA->android should see 10");

        let res = tree.predict(vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
            Feature::string("network", "wifi"),
        ]);
        assert_eq!(res, Ok(Some(10)), "Banner->wifi should see 10");

        // Verify all leaf combinations exist (2 formats × 2 sizes = 4 combinations)
        let combinations = vec![
            ("banner", "300x250"),
            ("banner", "728x90"),
            ("video", "300x250"),
            ("video", "728x90"),
        ];

        for (format, size) in combinations {
            let res = tree.predict(vec![
                Feature::string("country", "usa"),
                Feature::string("device", "android"),
                Feature::string("format", format),
                Feature::string("network", "wifi"),
                Feature::string("size", size),
            ]);
            assert_eq!(
                res,
                Ok(Some(10)),
                "Leaf node {}->wifi->{} should exist with value 10",
                format,
                size
            );
        }

        // Verify count: should have 4 leaf nodes (banner×2 sizes + video×2 sizes)
        let leaf_count = tree.size(true);
        assert_eq!(leaf_count, 4, "Should have 4 leaf nodes (2 formats × 2 sizes)");
    }
}
