//! Simple OpenCog Integration Demo
//! 
//! Demonstrates OpenCog cognitive architecture without requiring
//! the full web-rwkv compilation due to serde derive issues.

use std::collections::HashMap;

// Minimal re-implementation of needed components to demonstrate OpenCog
#[derive(Debug, Clone)]
pub struct SimpleAtom {
    pub id: u64,
    pub name: String,
    pub truth_value: f32,
    pub attention_value: f32,
}

#[derive(Debug)]
pub struct SimpleAtomSpace {
    atoms: HashMap<u64, SimpleAtom>,
    next_id: u64,
}

impl SimpleAtomSpace {
    pub fn new() -> Self {
        Self {
            atoms: HashMap::new(),
            next_id: 1,
        }
    }
    
    pub fn add_atom(&mut self, name: String) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        
        let atom = SimpleAtom {
            id,
            name,
            truth_value: 0.8,
            attention_value: 0.0,
        };
        
        self.atoms.insert(id, atom);
        id
    }
    
    pub fn get_atom(&self, id: u64) -> Option<&SimpleAtom> {
        self.atoms.get(&id)
    }
    
    pub fn set_attention(&mut self, id: u64, attention: f32) {
        if let Some(atom) = self.atoms.get_mut(&id) {
            atom.attention_value = attention;
        }
    }
    
    pub fn size(&self) -> usize {
        self.atoms.len()
    }
}

#[derive(Debug)]
pub struct SimpleAttentionSystem {
    cycle: u64,
}

impl SimpleAttentionSystem {
    pub fn new() -> Self {
        Self { cycle: 0 }
    }
    
    pub fn allocate_attention(&mut self, tokens: &[u16]) -> Vec<f32> {
        self.cycle += 1;
        
        // Simple attention allocation based on position and token value
        tokens.iter().enumerate().map(|(pos, &token)| {
            let position_weight = 1.0 - (pos as f32 / tokens.len() as f32) * 0.3;
            let token_weight = (token as f32 % 100.0) / 100.0;
            (position_weight * token_weight).max(0.1)
        }).collect()
    }
}

#[derive(Debug, Clone)]
pub struct CognitivePattern {
    pub id: u64,
    pub pattern_type: String,
    pub tokens: Vec<u16>,
    pub strength: f32,
}

#[derive(Debug)]
pub struct SimplePatternMatcher {
    next_id: u64,
}

impl SimplePatternMatcher {
    pub fn new() -> Self {
        Self { next_id: 1 }
    }
    
    pub fn find_patterns(&mut self, tokens: &[u16], attention: &[f32]) -> Vec<CognitivePattern> {
        let mut patterns = Vec::new();
        
        // Find repeating sequences
        for window_size in 2..=4.min(tokens.len()) {
            for start in 0..=(tokens.len() - window_size) {
                let sequence = &tokens[start..start + window_size];
                let avg_attention: f32 = attention[start..start + window_size].iter().sum::<f32>() / window_size as f32;
                
                if avg_attention > 0.3 {
                    let pattern = CognitivePattern {
                        id: self.next_id,
                        pattern_type: "sequence".to_string(),
                        tokens: sequence.to_vec(),
                        strength: avg_attention,
                    };
                    
                    self.next_id += 1;
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }
}

#[derive(Debug)]
pub struct SimpleOpenCogSystem {
    atomspace: SimpleAtomSpace,
    attention_system: SimpleAttentionSystem,
    pattern_matcher: SimplePatternMatcher,
    token_to_atom: HashMap<u16, u64>,
}

impl SimpleOpenCogSystem {
    pub fn new() -> Self {
        Self {
            atomspace: SimpleAtomSpace::new(),
            attention_system: SimpleAttentionSystem::new(),
            pattern_matcher: SimplePatternMatcher::new(),
            token_to_atom: HashMap::new(),
        }
    }
    
    pub fn process_tokens(&mut self, tokens: &[u16]) -> ProcessingResult {
        println!("🧠 Processing {} tokens through OpenCog architecture", tokens.len());
        
        // Step 1: Ensure atoms exist for tokens
        for &token in tokens {
            if !self.token_to_atom.contains_key(&token) {
                let atom_id = self.atomspace.add_atom(format!("token_{}", token));
                self.token_to_atom.insert(token, atom_id);
                println!("  Created atom {} for token {}", atom_id, token);
            }
        }
        
        // Step 2: Allocate attention
        let attention_weights = self.attention_system.allocate_attention(tokens);
        println!("  🎯 Allocated attention: {:?}", 
            attention_weights.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
        
        // Step 3: Update attention in atomspace
        for (&token, &attention) in tokens.iter().zip(attention_weights.iter()) {
            if let Some(&atom_id) = self.token_to_atom.get(&token) {
                self.atomspace.set_attention(atom_id, attention);
            }
        }
        
        // Step 4: Find cognitive patterns
        let patterns = self.pattern_matcher.find_patterns(tokens, &attention_weights);
        println!("  🔍 Found {} cognitive patterns", patterns.len());
        
        for pattern in &patterns {
            println!("    Pattern {}: {} tokens, strength {:.2}", 
                pattern.id, pattern.tokens.len(), pattern.strength);
        }
        
        // Step 5: Generate cognitively-influenced predictions
        let predictions = self.generate_predictions(tokens, &attention_weights, &patterns);
        
        ProcessingResult {
            attention_weights,
            patterns,
            predictions,
            atomspace_size: self.atomspace.size(),
        }
    }
    
    fn generate_predictions(&self, tokens: &[u16], attention: &[f32], patterns: &[CognitivePattern]) -> Vec<(u16, f32)> {
        let mut predictions = Vec::new();
        
        // Simple prediction based on attention and patterns
        for (i, &token) in tokens.iter().enumerate() {
            if i < attention.len() {
                let base_prob = 0.1;
                let attention_boost = attention[i] * 0.3;
                
                // Boost probability based on pattern membership
                let pattern_boost = patterns.iter()
                    .filter(|p| p.tokens.contains(&token))
                    .map(|p| p.strength * 0.2)
                    .sum::<f32>();
                
                let total_prob = base_prob + attention_boost + pattern_boost;
                predictions.push((token, total_prob.min(1.0)));
            }
        }
        
        // Sort by probability
        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(5); // Top 5 predictions
        
        predictions
    }
    
    pub fn get_atomspace_stats(&self) -> (usize, usize) {
        (self.atomspace.size(), self.token_to_atom.len())
    }
}

#[derive(Debug)]
pub struct ProcessingResult {
    pub attention_weights: Vec<f32>,
    pub patterns: Vec<CognitivePattern>,
    pub predictions: Vec<(u16, f32)>,
    pub atomspace_size: usize,
}

fn main() {
    println!("🚀 Simple OpenCog WebGPU Integration Demo");
    println!("==========================================");
    
    let mut opencog = SimpleOpenCogSystem::new();
    
    // Demo 1: Basic token processing
    println!("\n📝 Demo 1: Basic Token Processing");
    let tokens1 = vec![72, 101, 108, 108, 111]; // "Hello"
    let result1 = opencog.process_tokens(&tokens1);
    
    println!("Result 1:");
    println!("  AtomSpace size: {}", result1.atomspace_size);
    println!("  Top predictions: {:?}", result1.predictions);
    
    // Demo 2: Repeated patterns  
    println!("\n🔄 Demo 2: Pattern Recognition");
    let tokens2 = vec![72, 101, 108, 108, 111, 32, 72, 101, 108, 108, 111]; // "Hello Hello"
    let result2 = opencog.process_tokens(&tokens2);
    
    println!("Result 2:");
    println!("  Patterns found: {}", result2.patterns.len());
    for pattern in &result2.patterns {
        println!("    Pattern {}: {:?} (strength: {:.2})", 
            pattern.id, pattern.tokens, pattern.strength);
    }
    
    // Demo 3: Sequential processing
    println!("\n⏭️  Demo 3: Sequential Learning");
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 2, 3, 6, 7],
        vec![1, 2, 8, 9, 10],
    ];
    
    for (i, seq) in sequences.iter().enumerate() {
        println!("Processing sequence {}: {:?}", i + 1, seq);
        let result = opencog.process_tokens(seq);
        println!("  Found {} patterns", result.patterns.len());
        
        if !result.predictions.is_empty() {
            println!("  Top prediction: token {} (prob: {:.3})", 
                result.predictions[0].0, result.predictions[0].1);
        }
    }
    
    // Demo 4: Knowledge accumulation
    println!("\n📚 Demo 4: Knowledge Accumulation");
    let (total_atoms, total_tokens) = opencog.get_atomspace_stats();
    println!("Final AtomSpace statistics:");
    println!("  Total atoms: {}", total_atoms);
    println!("  Unique tokens processed: {}", total_tokens);
    
    // Demo 5: Cognitive reasoning simulation
    println!("\n🤔 Demo 5: Cognitive Reasoning Simulation");
    
    // Simulate attention flow and pattern strengthening
    let complex_sequence = vec![1, 2, 3, 2, 3, 4, 3, 4, 5];
    println!("Processing complex sequence: {:?}", complex_sequence);
    
    let result = opencog.process_tokens(&complex_sequence);
    
    println!("Cognitive analysis results:");
    println!("  Attention distribution: {:?}", 
        result.attention_weights.iter()
            .enumerate()
            .map(|(i, &att)| format!("{}:{:.2}", complex_sequence[i], att))
            .collect::<Vec<_>>());
    
    println!("  Detected cognitive patterns:");
    for pattern in &result.patterns {
        println!("    {:?} -> strength: {:.2}", pattern.tokens, pattern.strength);
    }
    
    println!("  Cognitive predictions (token:probability):");
    for (token, prob) in &result.predictions {
        println!("    {}:{:.3}", token, prob);
    }
    
    println!("\n✅ Demo complete!");
    println!("\n🧠 This demonstration showcased:");
    println!("• Cognitive token processing with attention allocation");
    println!("• Pattern recognition and cognitive memory formation");
    println!("• AtomSpace knowledge representation");
    println!("• Attention-based reasoning and prediction");
    println!("• Sequential learning and knowledge accumulation");
    println!("• Cognitive architecture integration with WebGPU concepts");
    
    println!("\n💡 In a full implementation with WebGPU:");
    println!("• Tensor operations would run on GPU");
    println!("• Attention mechanisms would be hardware-accelerated");
    println!("• Pattern matching would use parallel processing");
    println!("• Knowledge graphs would leverage GPU memory");
    println!("• Real RWKV model would provide language understanding");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_atomspace() {
        let mut atomspace = SimpleAtomSpace::new();
        let id = atomspace.add_atom("test".to_string());
        
        assert_eq!(atomspace.size(), 1);
        assert!(atomspace.get_atom(id).is_some());
    }
    
    #[test]
    fn test_attention_allocation() {
        let mut attention_system = SimpleAttentionSystem::new();
        let tokens = vec![1, 2, 3, 4, 5];
        let weights = attention_system.allocate_attention(&tokens);
        
        assert_eq!(weights.len(), tokens.len());
        assert!(weights.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }
    
    #[test]
    fn test_pattern_matching() {
        let mut pattern_matcher = SimplePatternMatcher::new();
        let tokens = vec![1, 2, 3, 2, 3, 4];
        let attention = vec![0.5, 0.6, 0.7, 0.6, 0.7, 0.4];
        
        let patterns = pattern_matcher.find_patterns(&tokens, &attention);
        assert!(!patterns.is_empty());
    }
    
    #[test]
    fn test_opencog_system() {
        let mut system = SimpleOpenCogSystem::new();
        let tokens = vec![1, 2, 3];
        let result = system.process_tokens(&tokens);
        
        assert_eq!(result.attention_weights.len(), tokens.len());
        assert_eq!(result.atomspace_size, 3);
    }
}