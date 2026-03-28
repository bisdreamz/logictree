mod common;

use common::{MaxValueHandler, SummingHandler};
use logictree::{Feature, LogicTree};

#[test]
fn test_fan_out_creates_independent_children() {
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
            Feature::string("device", "android"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ],
        &10,
    )
    .unwrap();

    // Parent trained once, not double-counted
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10, "Parent should not be double-counted");
    assert_eq!(res.depth, 2);

    // Each child trained independently
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);
    assert_eq!(res.depth, 3);

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "video"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);

    // Multi-value predict: both children equal (10), max picks either
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);
    assert_eq!(res.depth, 3);
    assert_eq!(res.full_depth, true);
}

#[test]
fn test_multi_value_device_fan_out() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        MaxValueHandler::new(),
    );

    // Fan-out on device level
    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::multi_string("device", vec!["android", "ios"]),
            Feature::string("format", "banner"),
        ],
        &5,
    )
    .unwrap();

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 5);

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ios"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 5);

    // Parent trained once
    let res = tree
        .predict(&vec![Feature::string("country", "usa")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 5, "Parent should see 5, not 10");
}

#[test]
fn test_ctv_accumulation() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
        ],
        MaxValueHandler::new(),
    );

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

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
            Feature::string("format", "video"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "ctv"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);
}

#[test]
fn test_empty_multi_value_returns_error() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    let result = tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", Vec::<&str>::new()),
        ],
        &10,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("cannot have empty values"));
}

#[test]
fn test_no_cross_contamination() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ],
        &10,
    )
    .unwrap();

    // Each child trained independently with 10
    let banner = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(banner.value, 10);

    let video = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(video.value, 10);
}

#[test]
fn test_serialization_preserves_fan_out() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ],
        &42,
    )
    .unwrap();

    let file = "test_multi_value_serialize.bin";
    tree.save(file).expect("Should serialize");

    let loaded: LogicTree<u32, u32, MaxValueHandler> =
        LogicTree::load(file).expect("Should deserialize");

    let res = loaded
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 42);

    std::fs::remove_file(file).expect("Should delete test file");
}

#[test]
fn test_summing_resolve() {
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

    // Multi-value sums both children
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 300, "Sum of banner (100) + video (200)");

    // Single-value returns individual child
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);
}

#[test]
fn test_summing_with_depth() {
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

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 301);
    assert_eq!(res.depth, 2);
    assert_eq!(res.full_depth, true);
}

#[test]
fn test_nested_multi_value_fan_out() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "device".to_string(),
            "format".to_string(),
            "network".to_string(),
            "size".to_string(),
        ],
        MaxValueHandler::new(),
    );

    // {banner,video} x {300x250,728x90} = 4 leaf paths
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

    let res = tree
        .predict(&vec![Feature::string("country", "usa")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10, "Parent trained once, not 4x");

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("device", "android"),
            Feature::multi_string("format", vec!["banner", "video"]),
            Feature::string("network", "wifi"),
            Feature::multi_string("size", vec!["300x250", "728x90"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 10);

    assert_eq!(tree.size(true), 4, "Should have 4 leaf nodes");
}

#[test]
fn test_fan_out_with_single_value_predict() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ],
        &100,
    )
    .unwrap();

    let banner = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(banner.value, 100);

    let video = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(video.value, 100);

    // Both equal — max picks either
    let multi = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(multi.value, 100);
}

#[test]
fn test_value_order_independence() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
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

    let banner = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(banner.value, 100);

    let video = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "video"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(video.value, 100);
}

#[test]
fn test_summing_accumulation() {
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

    // banner = 100 + 50 = 150, video = 200 + 50 = 250
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 400, "Sum of banner (150) + video (250)");
}

#[test]
fn test_overlapping_fan_outs() {
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

    // banner = 100 + 200 = 300, video = 100
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 400, "Sum of banner (300) + video (100)");
}

#[test]
fn test_partial_multi_query() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ],
        &100,
    )
    .unwrap();

    // Multi-value query where only banner exists — single match returned
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);
}

#[test]
fn test_deeper_children_with_fan_out() {
    let tree = LogicTree::new(
        vec![
            "country".to_string(),
            "format".to_string(),
            "size".to_string(),
        ],
        MaxValueHandler::new(),
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

    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::multi_string("format", vec!["banner", "video"]),
            Feature::string("size", "300x250"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);
}

#[test]
fn test_fan_out_accumulates_on_single_query() {
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

    // banner = 100 + 50 = 150
    let res = tree
        .predict(&vec![
            Feature::string("country", "usa"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 150);
    assert_eq!(res.depth, 2);
    assert_eq!(res.full_depth, true);
}

#[test]
fn test_consecutive_multi_value_features() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

    tree.train(
        &vec![
            Feature::multi_string("country", vec!["USA", "Canada"]),
            Feature::multi_string("format", vec!["banner", "video"]),
        ],
        &100,
    )
    .unwrap();

    let res = tree
        .predict(&vec![Feature::string("country", "USA")])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);

    let res = tree
        .predict(&vec![
            Feature::string("country", "USA"),
            Feature::string("format", "banner"),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);

    let res = tree
        .predict(&vec![
            Feature::string("country", "USA"),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);

    let res = tree
        .predict(&vec![
            Feature::multi_string("country", vec!["USA", "Canada"]),
            Feature::multi_string("format", vec!["banner", "video"]),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 100);
}

#[test]
fn test_owned_strings() {
    let tree = LogicTree::new(
        vec!["country".to_string(), "format".to_string()],
        MaxValueHandler::new(),
    );

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
    .expect("Should work with owned strings");

    let mixed_formats = vec!["banner".to_string(), "video".to_string()];
    let res = tree
        .predict(&vec![
            Feature::multi_string("country", vec!["USA", "Canada"]),
            Feature::multi_string("format", mixed_formats),
        ])
        .unwrap()
        .unwrap();
    assert_eq!(res.value, 42);

    // Iterator input
    let iter_formats = vec!["banner", "native"].into_iter();
    let _ = tree.predict(&vec![
        Feature::string("country", "USA"),
        Feature::multi_string("format", iter_formats),
    ]);
}
