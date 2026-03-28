mod common;

use common::{MaxValueHandler, ThresholdHandler};
use logictree::{Feature, LogicTree};

#[test]
fn test_insufficient_child_falls_back_to_parent() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string()],
        ThresholdHandler::new(10),
    );

    // Parent gets 20 samples — above threshold
    for _ in 0..20 {
        tree.train(&vec![Feature::string("country", "usa")], &1)
            .unwrap();
    }

    // Child gets only 3 — below threshold
    for _ in 0..3 {
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("device", "ios"),
            ],
            &100,
        )
        .unwrap();
    }

    let pred = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(pred.depth, 1, "Should fall back to parent at depth 1");
    // Parent total = 20*1 + 3*100 = 320
    assert_eq!(pred.value, 320);
}

#[test]
fn test_sufficient_child_returns_at_child_depth() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "device".to_string()],
        ThresholdHandler::new(10),
    );

    for _ in 0..20 {
        tree.train(&vec![Feature::string("country", "usa")], &1)
            .unwrap();
    }

    // Child gets 15 samples — above threshold
    for _ in 0..15 {
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("device", "ios"),
            ],
            &100,
        )
        .unwrap();
    }

    let pred = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(pred.depth, 2, "Sufficient child returns at depth 2");
    assert_eq!(pred.value, 1500);
}

#[test]
fn test_single_child_must_not_bypass_resolve() {
    // Validates that resolve() is always called, even with a single matching child.
    // Without this, under-trained deep paths would leak through without sufficiency checks.
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        ThresholdHandler::new(10),
    );

    for _ in 0..20 {
        tree.train(&vec![Feature::string("country", "usa")], &1)
            .unwrap();
    }

    // Deep path with only 2 samples — each level has exactly one matching child
    for _ in 0..2 {
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::string("device", "ios"),
                Feature::string("format", "banner"),
            ],
            &999,
        )
        .unwrap();
    }

    let pred = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();

    assert_eq!(
        pred.depth, 1,
        "Should fall back to country=usa (depth 1), not leak through to depth 3"
    );
}

#[test]
fn test_insufficient_multi_value_falls_back() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        ThresholdHandler::new(10),
    );

    for _ in 0..20 {
        tree.train(&vec![Feature::string("country", "usa")], &1)
            .unwrap();
    }

    // Both format children get insufficient samples
    for _ in 0..3 {
        tree.train(
            &vec![
                Feature::string("country", "usa"),
                Feature::multi_string("format", vec!["banner", "video"]),
            ],
            &100,
        )
        .unwrap();
    }

    let pred = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();

    assert_eq!(pred.depth, 1, "Should fall back to parent");
}

#[test]
fn test_depth_reports_selected_prediction() {
    // MaxValueHandler picks the highest value. When the highest-value prediction
    // comes from a shallower depth, the reported depth must match.
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "format".to_string(),
            "size".to_string(),
        ],
        MaxValueHandler::new(),
    );

    // Train banner to depth 2 with HIGH value (no size child)
    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ],
        &500,
    )
    .unwrap();

    // Train video to depth 3 with LOW value
    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
            Feature::string("size", "640x480"),
        ],
        &200,
    )
    .unwrap();

    // Multi-value predict: banner falls back to depth 2 (value 500),
    // video reaches depth 3 (value 200). MaxValueHandler picks banner.
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
            Feature::string("size", "640x480"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(
        res.value, 500,
        "MaxValueHandler picks banner (higher value)"
    );
    assert_eq!(
        res.depth, 2,
        "Should report banner's depth 2, not video's depth 3"
    );
}
