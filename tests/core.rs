mod common;

use common::MaxValueHandler;
use logictree::{Feature, LogicTree, PredictionHandler};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Mutex;

#[test]
fn test_train_and_predict() {
    let tree = LogicTree::new(
        vec!["one".to_string(), "two".to_string()],
        MaxValueHandler::new(),
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

    // Parent prediction: foo has total=20 (10+5+5)
    let res = tree
        .predict(&vec![Feature::string("one", "foo")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 20);
    assert_eq!(res.depth, 1);

    // Child predictions
    let res = tree
        .predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bar"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);
    assert_eq!(res.depth, 2);
    assert_eq!(res.full_depth, true);

    let res = tree
        .predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "baz"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 5);

    let res = tree
        .predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bum"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 5);

    // Unknown path falls back to root
    let res = tree
        .predict(&vec![
            Feature::string("one", "notfoo"),
            Feature::string("two", "baz"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 20, "Unknown path falls back to root");
    assert_eq!(res.depth, 0);

    // Partial unknown path falls back to root
    let res = tree
        .predict(&vec![Feature::string("one", "notfoo")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 20);
    assert_eq!(res.depth, 0);

    // Empty features returns root prediction
    let res = tree.predict(&vec![]).unwrap().unwrap();
    assert_eq!(res.value, 20);
    assert_eq!(res.depth, 0);
}

#[test]
fn test_invalid_feature_order() {
    let tree = LogicTree::new(
        vec!["one".to_string(), "two".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![Feature::string("one", "foo"), Feature::string("two", "bar")],
        &10,
    )
    .unwrap();

    tree.train(
        &vec![Feature::string("two", "baz"), Feature::string("one", "foo")],
        &10,
    )
    .expect_err("Should fail on invalid feature order");
}

#[test]
fn test_tree_size() {
    let tree = LogicTree::new(
        vec!["one".to_string(), "two".to_string()],
        MaxValueHandler::new(),
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

    assert_eq!(tree.size(true), 3, "Should count 3 leaf nodes");
    assert_eq!(tree.size(false), 5, "Should count 5 total nodes");
}

#[test]
fn test_serialization_roundtrip() {
    let tree = LogicTree::new(
        vec!["one".to_string(), "two".to_string()],
        MaxValueHandler::new(),
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

    let file = "test_core_serialization.bin";
    tree.save(file).expect("Should serialize");

    let loaded: LogicTree<u32, u32, MaxValueHandler> =
        LogicTree::load(file).expect("Should deserialize");

    let res = loaded
        .predict(&vec![Feature::string("one", "foo")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 15, "Loaded tree preserves predictions");

    let res = loaded
        .predict(&vec![
            Feature::string("one", "foo"),
            Feature::string("two", "bar"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);

    std::fs::remove_file(file).expect("Should delete test file");
}

#[test]
fn test_prune() {
    #[derive(Serialize, Deserialize)]
    struct PrunableHandler {
        total: Mutex<u32>,
    }

    impl Clone for PrunableHandler {
        fn clone(&self) -> Self {
            PrunableHandler {
                total: Mutex::new(*self.total.lock().unwrap()),
            }
        }
    }

    impl PrunableHandler {
        fn new() -> Self {
            PrunableHandler {
                total: Mutex::new(0),
            }
        }
    }

    impl PredictionHandler<u32, u32> for PrunableHandler {
        fn train(&self, input: &u32, _next: Option<&Feature>) {
            *self.total.lock().unwrap() += input;
        }

        fn predict(&self) -> u32 {
            *self.total.lock().unwrap()
        }

        fn should_prune(&self) -> bool {
            true
        }

        fn new_instance(&self) -> Self {
            PrunableHandler::new()
        }

        fn resolve(&self, predictions: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> {
            predictions.into_iter().max_by_key(|(v, _)| *v)
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

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);

    assert!(tree.prune(), "Should prune entire tree");
}

// Depth tracking tests

#[test]
fn test_depth_at_root() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(&vec![], &10).unwrap();

    let res = tree.predict(&vec![]).unwrap().unwrap();
    assert_eq!(res.depth, 0);
    assert_eq!(res.full_depth, false);
    assert_eq!(res.value, 10);
}

#[test]
fn test_depth_at_level_1() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        MaxValueHandler::new(),
    );

    tree.train(&vec![Feature::string("country", "usa")], &20)
        .unwrap();

    let res = tree
        .predict(&vec![Feature::string("country", "usa")])
        .unwrap()
        .unwrap();
    assert_eq!(res.depth, 1);
    assert_eq!(res.full_depth, false);
    assert_eq!(res.value, 20);
}

#[test]
fn test_depth_at_full_depth() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ],
        &30,
    )
    .unwrap();

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.depth, 2);
    assert_eq!(res.full_depth, true);
    assert_eq!(res.value, 30);
}

#[test]
fn test_depth_fallback_to_parent() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        MaxValueHandler::new(),
    );

    tree.train(&vec![Feature::string("country", "usa")], &15)
        .unwrap();

    // Unknown device — falls back to country=usa
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "unknown"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.depth, 1);
    assert_eq!(res.full_depth, false);
    assert_eq!(res.value, 15);
}

#[test]
fn test_depth_empty_tree() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string()],
        MaxValueHandler::new(),
    );

    let res = tree.predict(&vec![]).unwrap().unwrap();
    assert_eq!(res.depth, 0);
    assert_eq!(res.full_depth, false);
    assert_eq!(res.value, 0);
}

#[test]
fn test_depth_partial_path() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ],
        &25,
    )
    .unwrap();

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.depth, 2);
    assert_eq!(res.full_depth, false, "Not full depth for 3-feature tree");
    assert_eq!(res.value, 25);
}

// Validation tests

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
