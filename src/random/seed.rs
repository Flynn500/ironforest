use crate::random::Generator;

pub struct SeedSequence {
    entropy: u64,
    spawn_count: u64,
}

impl SeedSequence {
    pub fn new(seed: u64) -> Self {
        Self {
            entropy: seed,
            spawn_count: 0,
        }
    }

    pub fn spawn(&mut self) -> Self {
        let child_entropy = self.mix(self.spawn_count);
        self.spawn_count += 1;

        Self {
            entropy: child_entropy,
            spawn_count: 0,
        }
    }

    fn mix(&self, index: u64) -> u64 {
        let mut z = self.entropy.wrapping_add(index.wrapping_mul(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn into_generator(self) -> Generator {
        Generator::from_seed(self.entropy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_is_deterministic() {
        let mut seq1 = SeedSequence::new(12345);
        let mut seq2 = SeedSequence::new(12345);

        let child1a = seq1.spawn();
        let child1b = seq2.spawn();

        assert_eq!(child1a.entropy, child1b.entropy);
    }

    #[test]
    fn spawned_children_differ() {
        let mut seq = SeedSequence::new(12345);

        let child1 = seq.spawn();
        let child2 = seq.spawn();

        assert_ne!(child1.entropy, child2.entropy);
    }

    #[test]
    fn generators_from_spawns_are_independent() {
        let mut seq = SeedSequence::new(99999);

        let mut rng1 = seq.spawn().into_generator();
        let mut rng2 = seq.spawn().into_generator();

        // Should produce different sequences
        let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

        assert_ne!(vals1, vals2);
    }

    #[test]
    fn spawn_resets_child_count() {
        let mut seq = SeedSequence::new(12345);
        let mut child = seq.spawn();

        assert_eq!(child.spawn_count, 0);
    }
}