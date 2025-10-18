//! OpenCog WebGPU Inference Demo
//! 
//! Demonstrates the integration of OpenCog cognitive architecture
//! with the RWKV WebGPU inference engine.

use web_rwkv::opencog::{
    OpenCogSystem, 
    CognitiveInferenceEngine, 
    RwkvIntegrationParams,
    AtomType,
    TruthValue,
};

fn main() -> anyhow::Result<()> {
    simple_logger::SimpleLogger::new().init().unwrap();
    
    println!("OpenCog WebGPU Inference Demo");
    println!("============================");
    
    // Initialize the cognitive system
    let mut opencog_system = OpenCogSystem::new();
    println!("✓ OpenCog system initialized");
    
    // Initialize the cognitive inference engine
    let mut inference_engine = CognitiveInferenceEngine::new();
    println!("✓ Cognitive inference engine initialized");
    
    // Set up integration parameters
    let params = RwkvIntegrationParams {
        embedding_dim: 768,
        sequence_length: 512,
        attention_threshold: 0.15,
        pattern_sensitivity: 0.4,
        learning_rate: 0.02,
    };
    inference_engine.set_params(params);
    println!("✓ Integration parameters configured");
    
    // Demo 1: Process sample text tokens
    println!("\n--- Demo 1: Token Processing ---");
    let sample_tokens = vec![72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]; // "Hello world" in ASCII
    println!("Processing tokens: {:?}", sample_tokens);
    
    let cognitive_result = inference_engine.process_tokens(&sample_tokens);
    println!("✓ Cognitive processing completed");
    println!("  - Patterns detected: {}", cognitive_result.patterns.len());
    println!("  - Knowledge updates: {}", if cognitive_result.knowledge_updates.is_empty() { "none" } else { "yes" });
    println!("  - Reasoning steps: {}", cognitive_result.reasoning_results.steps.len());
    println!("  - Confidence: {:.3}", cognitive_result.confidence);
    
    // Demo 2: Attention allocation
    println!("\n--- Demo 2: Attention Allocation ---");
    println!("Attention weights: {:?}", 
        cognitive_result.cognitive_attention.iter()
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
    
    let attention_focus = inference_engine.get_attention_focus();
    println!("Attentional focus: {} atoms", attention_focus.len());
    
    // Demo 3: Pattern analysis
    println!("\n--- Demo 3: Pattern Analysis ---");
    for (i, pattern) in cognitive_result.patterns.iter().enumerate() {
        println!("  Pattern {}: type={:?}, strength={:.3}, atoms={}", 
            i + 1, 
            pattern.pattern_type, 
            pattern.strength,
            pattern.atoms.len()
        );
    }
    
    // Demo 4: Knowledge querying
    println!("\n--- Demo 4: Knowledge Querying ---");
    let query_tokens = vec![72, 101, 108, 108, 111]; // "Hello"
    let related_patterns = inference_engine.query_knowledge(&query_tokens);
    println!("Found {} patterns related to query", related_patterns.len());
    
    // Demo 5: Simple text generation
    println!("\n--- Demo 5: Cognitive Text Generation ---");
    let context = vec![72, 101, 108, 108, 111]; // Start with "Hello"
    println!("Context: {:?}", context);
    
    if let Some(next_token) = inference_engine.generate_next_token(&context) {
        println!("Next token prediction: {}", next_token);
    } else {
        println!("No confident prediction available");
    }
    
    // Demo 6: AtomSpace operations
    println!("\n--- Demo 6: AtomSpace Operations ---");
    let mut atomspace = opencog_system.atomspace;
    
    // Create some example atoms
    let concept_hello = atomspace.add_node(AtomType::ConceptNode, "hello".to_string());
    let concept_world = atomspace.add_node(AtomType::ConceptNode, "world".to_string());
    let greeting_link = atomspace.add_link(AtomType::EvaluationLink, vec![concept_hello, concept_world]);
    
    println!("Created atoms: hello={}, world={}, link={}", concept_hello, concept_world, greeting_link);
    
    // Set truth values
    atomspace.set_truth_value(concept_hello, TruthValue::certain(0.9));
    atomspace.set_truth_value(concept_world, TruthValue::certain(0.8));
    
    println!("AtomSpace size: {} atoms", atomspace.size());
    
    // Demo 7: Processing statistics
    println!("\n--- Demo 7: Processing Statistics ---");
    let stats = inference_engine.get_stats();
    println!("  Tokens processed: {}", stats.tokens_processed);
    println!("  Patterns detected: {}", stats.patterns_detected);
    println!("  Knowledge updates: {}", stats.knowledge_updates);
    println!("  Reasoning steps: {}", stats.reasoning_steps);
    println!("  Average confidence: {:.3}", stats.avg_confidence);
    
    // Demo 8: Knowledge export
    println!("\n--- Demo 8: Knowledge Export ---");
    let knowledge_export = inference_engine.export_knowledge();
    println!("  Total atoms: {}", knowledge_export.total_atoms);
    println!("  Total patterns: {}", knowledge_export.total_patterns);
    println!("  Attention focus size: {}", knowledge_export.attention_focus.len());
    println!("  Knowledge nodes: {}", knowledge_export.knowledge_stats.total_nodes);
    println!("  Knowledge connections: {}", knowledge_export.knowledge_stats.total_connections);
    
    // Demo 9: Cognitive reasoning example
    println!("\n--- Demo 9: Cognitive Reasoning ---");
    
    // Process multiple related sequences to build knowledge
    let sequences = vec![
        vec![72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100], // "Hello world"
        vec![72, 101, 108, 108, 111, 32, 116, 104, 101, 114, 101], // "Hello there"
        vec![72, 105, 32, 116, 104, 101, 114, 101], // "Hi there"
    ];
    
    for (i, sequence) in sequences.iter().enumerate() {
        println!("Processing sequence {}: {:?}", i + 1, sequence);
        let result = inference_engine.process_tokens(sequence);
        
        if !result.reasoning_results.steps.is_empty() {
            println!("  Reasoning: {} steps performed", result.reasoning_results.steps.len());
            for step in &result.reasoning_results.steps {
                println!("    - {:?}: confidence {:.3}", step.reasoning_type, step.confidence);
            }
        }
    }
    
    // Demo 10: System integration
    println!("\n--- Demo 10: System Integration ---");
    
    // Process input through the complete OpenCog system
    let test_input = vec![84, 101, 115, 116]; // "Test"
    let cognitive_result = opencog_system.process(&test_input);
    
    println!("Complete cognitive processing:");
    println!("  Attention weights: {}", cognitive_result.attention_weights.len());
    println!("  Patterns found: {}", cognitive_result.patterns.len());
    
    println!("\n✓ OpenCog WebGPU Integration Demo Complete!");
    println!("\nThis demo showcased:");
    println!("• Token processing with cognitive attention");
    println!("• Pattern recognition and analysis");  
    println!("• Knowledge graph construction");
    println!("• Cognitive reasoning chains");
    println!("• AtomSpace knowledge representation");
    println!("• Attention allocation mechanisms");
    println!("• Integration statistics and monitoring");
    println!("• Cognitive-enhanced text generation");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_opencog_system_creation() {
        let system = OpenCogSystem::new();
        assert_eq!(system.atomspace.size(), 0);
    }
    
    #[test]
    fn test_cognitive_inference_engine() {
        let mut engine = CognitiveInferenceEngine::new();
        let tokens = vec![1, 2, 3, 4, 5];
        let result = engine.process_tokens(&tokens);
        
        assert_eq!(result.cognitive_attention.len(), tokens.len());
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }
    
    #[test]
    fn test_atomspace_operations() {
        let mut atomspace = web_rwkv::opencog::AtomSpace::new();
        
        let concept = atomspace.add_node(AtomType::ConceptNode, "test".to_string());
        assert_eq!(atomspace.size(), 1);
        
        let link = atomspace.add_link(AtomType::InheritanceLink, vec![concept]);
        assert_eq!(atomspace.size(), 2);
        
        atomspace.set_truth_value(concept, TruthValue::certain(0.8));
        let atom = atomspace.get_atom(concept).unwrap();
        assert_eq!(atom.truth_value().strength, 0.8);
    }
    
    #[test]
    fn test_attention_allocation() {
        let mut attention_system = web_rwkv::opencog::AttentionSystem::new();
        let input = vec![1, 2, 3, 4, 5];
        let weights = attention_system.allocate(&input);
        
        assert_eq!(weights.len(), input.len());
        
        // Check that weights sum to approximately 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_pattern_matching() {
        let mut pattern_matcher = web_rwkv::opencog::PatternMatcher::new();
        let tokens = vec![1, 2, 3, 2, 3, 4];
        let attention = vec![0.2, 0.3, 0.4, 0.3, 0.4, 0.1];
        
        let patterns = pattern_matcher.match_patterns(&tokens, &attention);
        assert!(!patterns.is_empty());
    }
}