use logictree::{Feature, PredictionHandler};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Mutex;

/// Selects the prediction with the highest value.
/// Represents "pick the best-performing segment."
#[derive(Serialize, Deserialize)]
pub struct MaxValueHandler {
    total: Mutex<u32>,
}

impl Clone for MaxValueHandler {
    fn clone(&self) -> Self {
        MaxValueHandler {
            total: Mutex::new(*self.total.lock().unwrap()),
        }
    }
}

impl MaxValueHandler {
    pub fn new() -> Self {
        MaxValueHandler {
            total: Mutex::new(0),
        }
    }
}

impl PredictionHandler<u32, u32> for MaxValueHandler {
    fn train(&self, input: &u32, _next: Option<&Feature>) {
        *self.total.lock().unwrap() += input;
    }

    fn predict(&self) -> u32 {
        *self.total.lock().unwrap()
    }

    fn should_prune(&self) -> bool {
        false
    }

    fn new_instance(&self) -> Self {
        MaxValueHandler::new()
    }

    fn resolve(&self, predictions: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> {
        predictions.into_iter().max_by_key(|(v, _)| *v)
    }
}

/// Sums all predictions from child nodes.
/// Represents "total revenue across all matching segments."
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub struct SummingHandler {
    total: Mutex<u32>,
}

impl Clone for SummingHandler {
    fn clone(&self) -> Self {
        SummingHandler {
            total: Mutex::new(*self.total.lock().unwrap()),
        }
    }
}

#[allow(dead_code)]
impl SummingHandler {
    pub fn new() -> Self {
        SummingHandler {
            total: Mutex::new(0),
        }
    }
}

impl PredictionHandler<u32, u32> for SummingHandler {
    fn train(&self, input: &u32, _next: Option<&Feature>) {
        *self.total.lock().unwrap() += input;
    }

    fn predict(&self) -> u32 {
        *self.total.lock().unwrap()
    }

    fn should_prune(&self) -> bool {
        false
    }

    fn new_instance(&self) -> Self {
        SummingHandler::new()
    }

    fn resolve(&self, predictions: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> {
        if predictions.is_empty() {
            return None;
        }
        let max_depth = predictions.iter().map(|(_, d)| *d).max().unwrap();
        Some((predictions.iter().map(|(v, _)| v).sum(), max_depth))
    }
}

/// Handler with a minimum sample threshold — returns None from resolve()
/// when the node hasn't seen enough training events, forcing fallback to parent.
/// Mimics real-world "don't trust predictions with insufficient data" pattern.
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub struct ThresholdHandler {
    count: Mutex<u32>,
    total: Mutex<u32>,
    min_samples: u32,
}

impl Clone for ThresholdHandler {
    fn clone(&self) -> Self {
        ThresholdHandler {
            count: Mutex::new(*self.count.lock().unwrap()),
            total: Mutex::new(*self.total.lock().unwrap()),
            min_samples: self.min_samples,
        }
    }
}

#[allow(dead_code)]
impl ThresholdHandler {
    pub fn new(min_samples: u32) -> Self {
        ThresholdHandler {
            count: Mutex::new(0),
            total: Mutex::new(0),
            min_samples,
        }
    }
}

impl PredictionHandler<u32, u32> for ThresholdHandler {
    fn train(&self, input: &u32, _next: Option<&Feature>) {
        *self.count.lock().unwrap() += 1;
        *self.total.lock().unwrap() += input;
    }

    fn predict(&self) -> u32 {
        *self.total.lock().unwrap()
    }

    fn should_prune(&self) -> bool {
        false
    }

    fn new_instance(&self) -> Self {
        ThresholdHandler::new(self.min_samples)
    }

    fn resolve(&self, predictions: SmallVec<[(u32, usize); 1]>) -> Option<(u32, usize)> {
        if *self.count.lock().unwrap() < self.min_samples {
            return None;
        }
        predictions.into_iter().max_by_key(|(v, _)| *v)
    }
}
