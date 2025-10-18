//! Knowledge Representation and Management for OpenCog
//! 
//! Implements knowledge graph structures and management for representing
//! and updating knowledge gained from RWKV model inference.

use crate::opencog::atomspace::{AtomSpace, AtomId, AtomType, TruthValue};
use crate::opencog::patterns::{CognitivePattern, PatternType};
use std::collections::{HashMap, HashSet, VecDeque};

/// Knowledge node representing a concept or entity
#[derive(Debug, Clone)]
pub struct KnowledgeNode {
    pub id: AtomId,
    pub concept: String,
    pub confidence: f32,
    pub activation: f32,
    pub connections: Vec<KnowledgeConnection>,
}

/// Connection between knowledge nodes
#[derive(Debug, Clone)]
pub struct KnowledgeConnection {
    pub target_id: AtomId,
    pub relation_type: RelationType,
    pub strength: f32,
    pub frequency: u32,
}

/// Types of relationships between knowledge nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    /// Inheritance relationship (is-a)
    Inheritance,
    /// Similarity relationship
    Similarity,
    /// Causal relationship
    Causality,
    /// Sequential relationship (temporal)
    Sequence,
    /// Part-of relationship
    PartOf,
    /// Synonym relationship
    Synonym,
    /// Antonym relationship
    Antonym,
    /// Context relationship
    Context,
}

/// Knowledge update from pattern integration
#[derive(Debug, Clone)]
pub struct KnowledgeUpdate {
    /// New concepts discovered
    pub new_concepts: Vec<String>,
    /// New relationships discovered
    pub new_relations: Vec<(AtomId, AtomId, RelationType, f32)>,
    /// Updated confidence values
    pub confidence_updates: Vec<(AtomId, f32)>,
    /// Activation changes
    pub activation_updates: Vec<(AtomId, f32)>,
}

impl KnowledgeUpdate {
    pub fn new() -> Self {
        Self {
            new_concepts: Vec::new(),
            new_relations: Vec::new(),
            confidence_updates: Vec::new(),
            activation_updates: Vec::new(),
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.new_concepts.is_empty() 
            && self.new_relations.is_empty()
            && self.confidence_updates.is_empty()
            && self.activation_updates.is_empty()
    }
}

impl Default for KnowledgeUpdate {
    fn default() -> Self {
        Self::new()
    }
}

/// Knowledge graph for storing and managing cognitive knowledge
#[derive(Debug)]
pub struct KnowledgeGraph {
    /// Knowledge nodes indexed by atom ID
    nodes: HashMap<AtomId, KnowledgeNode>,
    /// Concept to atom ID mapping
    concept_map: HashMap<String, AtomId>,
    /// Relationship index for efficient queries
    relation_index: HashMap<RelationType, Vec<(AtomId, AtomId)>>,
    /// Activation spreading parameters
    spreading_params: SpreadingParams,
    /// Recent activations for temporal decay
    activation_history: VecDeque<(AtomId, f32, u64)>,
}

/// Parameters for activation spreading
#[derive(Debug, Clone)]
pub struct SpreadingParams {
    /// Base spreading rate
    pub spread_rate: f32,
    /// Decay rate for activations
    pub decay_rate: f32,
    /// Maximum spreading distance
    pub max_distance: usize,
    /// Minimum activation threshold
    pub min_threshold: f32,
}

impl Default for SpreadingParams {
    fn default() -> Self {
        Self {
            spread_rate: 0.1,
            decay_rate: 0.95,
            max_distance: 3,
            min_threshold: 0.01,
        }
    }
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            concept_map: HashMap::new(),
            relation_index: HashMap::new(),
            spreading_params: SpreadingParams::default(),
            activation_history: VecDeque::new(),
        }
    }
    
    /// Integrate patterns into knowledge graph
    pub fn integrate(&mut self, patterns: &[CognitivePattern]) -> KnowledgeUpdate {
        let mut update = KnowledgeUpdate::new();
        
        for pattern in patterns {
            match pattern.pattern_type {
                PatternType::Sequence => {
                    self.integrate_sequence_pattern(pattern, &mut update);
                },
                PatternType::Semantic => {
                    self.integrate_semantic_pattern(pattern, &mut update);
                },
                PatternType::Attention => {
                    self.integrate_attention_pattern(pattern, &mut update);
                },
                PatternType::Causal => {
                    self.integrate_causal_pattern(pattern, &mut update);
                },
                _ => {
                    // Handle other pattern types
                    self.integrate_generic_pattern(pattern, &mut update);
                }
            }
        }
        
        // Perform activation spreading
        self.spread_activation();
        
        // Apply temporal decay
        self.apply_temporal_decay();
        
        update
    }
    
    /// Integrate sequential pattern into knowledge
    fn integrate_sequence_pattern(&mut self, pattern: &CognitivePattern, update: &mut KnowledgeUpdate) {
        // Create sequence relationships between consecutive atoms
        for window in pattern.atoms.windows(2) {
            let from_id = window[0];
            let to_id = window[1];
            
            self.add_or_strengthen_relation(
                from_id, 
                to_id, 
                RelationType::Sequence, 
                pattern.strength,
                update
            );
        }
    }
    
    /// Integrate semantic pattern into knowledge  
    fn integrate_semantic_pattern(&mut self, pattern: &CognitivePattern, update: &mut KnowledgeUpdate) {
        // Create similarity relationships between semantically related atoms
        for i in 0..pattern.atoms.len() {
            for j in (i + 1)..pattern.atoms.len() {
                let id1 = pattern.atoms[i];
                let id2 = pattern.atoms[j];
                
                self.add_or_strengthen_relation(
                    id1,
                    id2,
                    RelationType::Similarity,
                    pattern.strength * 0.5, // Weaker than direct sequence
                    update
                );
            }
        }
    }
    
    /// Integrate attention pattern into knowledge
    fn integrate_attention_pattern(&mut self, pattern: &CognitivePattern, update: &mut KnowledgeUpdate) {
        // Boost activation for atoms in attention pattern
        for &atom_id in &pattern.atoms {
            self.boost_activation(atom_id, pattern.strength, update);
        }
        
        // Create context relationships
        for window in pattern.atoms.windows(2) {
            let from_id = window[0];
            let to_id = window[1];
            
            self.add_or_strengthen_relation(
                from_id,
                to_id,
                RelationType::Context,
                pattern.attention_weight,
                update
            );
        }
    }
    
    /// Integrate causal pattern into knowledge
    fn integrate_causal_pattern(&mut self, pattern: &CognitivePattern, update: &mut KnowledgeUpdate) {
        // Create causal relationships
        for window in pattern.atoms.windows(2) {
            let cause_id = window[0];
            let effect_id = window[1];
            
            self.add_or_strengthen_relation(
                cause_id,
                effect_id,
                RelationType::Causality,
                pattern.strength,
                update
            );
        }
    }
    
    /// Integrate generic pattern
    fn integrate_generic_pattern(&mut self, pattern: &CognitivePattern, update: &mut KnowledgeUpdate) {
        // Basic integration - boost activation and create weak similarity links
        for &atom_id in &pattern.atoms {
            self.boost_activation(atom_id, pattern.strength * 0.3, update);
        }
    }
    
    /// Add or strengthen a relationship between atoms
    fn add_or_strengthen_relation(
        &mut self, 
        from_id: AtomId, 
        to_id: AtomId, 
        relation_type: RelationType,
        strength: f32,
        update: &mut KnowledgeUpdate
    ) {
        // Find or create connection
        let node = self.nodes.entry(from_id).or_insert_with(|| KnowledgeNode {
            id: from_id,
            concept: format!("concept_{}", from_id),
            confidence: 0.5,
            activation: 0.0,
            connections: Vec::new(),
        });
        
        // Look for existing connection
        if let Some(connection) = node.connections.iter_mut().find(|c| c.target_id == to_id && c.relation_type == relation_type) {
            // Strengthen existing connection
            connection.strength = (connection.strength + strength * 0.1).min(1.0);
            connection.frequency += 1;
        } else {
            // Add new connection
            let connection = KnowledgeConnection {
                target_id: to_id,
                relation_type: relation_type.clone(),
                strength,
                frequency: 1,
            };
            node.connections.push(connection);
            
            // Update relation index
            self.relation_index.entry(relation_type.clone()).or_default().push((from_id, to_id));
            
            update.new_relations.push((from_id, to_id, relation_type, strength));
        }
        
        // Ensure target node exists
        self.nodes.entry(to_id).or_insert_with(|| KnowledgeNode {
            id: to_id,
            concept: format!("concept_{}", to_id),
            confidence: 0.5,
            activation: 0.0,
            connections: Vec::new(),
        });
    }
    
    /// Boost activation for a node
    fn boost_activation(&mut self, atom_id: AtomId, boost: f32, update: &mut KnowledgeUpdate) {
        let node = self.nodes.entry(atom_id).or_insert_with(|| KnowledgeNode {
            id: atom_id,
            concept: format!("concept_{}", atom_id),
            confidence: 0.5,
            activation: 0.0,
            connections: Vec::new(),
        });
        
        let old_activation = node.activation;
        node.activation = (node.activation + boost).min(1.0);
        
        if (node.activation - old_activation).abs() > 0.01 {
            update.activation_updates.push((atom_id, node.activation));
            
            // Record in history for temporal decay
            self.activation_history.push_back((atom_id, boost, 0)); // 0 = current time
        }
    }
    
    /// Spread activation through the knowledge graph
    fn spread_activation(&mut self) {
        let mut new_activations: HashMap<AtomId, f32> = HashMap::new();
        
        // Spread from highly activated nodes
        for (&node_id, node) in &self.nodes {
            if node.activation > self.spreading_params.min_threshold {
                for connection in &node.connections {
                    let spread_amount = node.activation 
                        * connection.strength 
                        * self.spreading_params.spread_rate;
                    
                    *new_activations.entry(connection.target_id).or_insert(0.0) += spread_amount;
                }
            }
        }
        
        // Apply new activations
        for (node_id, activation_boost) in new_activations {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.activation = (node.activation + activation_boost).min(1.0);
            }
        }
    }
    
    /// Apply temporal decay to activations
    fn apply_temporal_decay(&mut self) {
        for node in self.nodes.values_mut() {
            node.activation *= self.spreading_params.decay_rate;
        }
        
        // Clean up old activation history
        let max_history_size = 1000;
        while self.activation_history.len() > max_history_size {
            self.activation_history.pop_front();
        }
    }
    
    /// Query related concepts
    pub fn query_related(&self, atom_id: AtomId, relation_type: Option<RelationType>) -> Vec<(AtomId, f32)> {
        if let Some(node) = self.nodes.get(&atom_id) {
            let mut related = Vec::new();
            
            for connection in &node.connections {
                if relation_type.is_none() || relation_type.as_ref() == Some(&connection.relation_type) {
                    related.push((connection.target_id, connection.strength));
                }
            }
            
            // Sort by strength
            related.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            related
        } else {
            Vec::new()
        }
    }
    
    /// Get most activated nodes
    pub fn get_most_activated(&self, limit: usize) -> Vec<&KnowledgeNode> {
        let mut nodes: Vec<&KnowledgeNode> = self.nodes.values().collect();
        nodes.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal));
        nodes.into_iter().take(limit).collect()
    }
    
    /// Get knowledge statistics
    pub fn get_stats(&self) -> KnowledgeStats {
        let total_nodes = self.nodes.len();
        let total_connections: usize = self.nodes.values().map(|n| n.connections.len()).sum();
        let avg_activation = if total_nodes > 0 {
            self.nodes.values().map(|n| n.activation).sum::<f32>() / total_nodes as f32
        } else {
            0.0
        };
        
        let mut relation_counts = HashMap::new();
        for node in self.nodes.values() {
            for connection in &node.connections {
                *relation_counts.entry(connection.relation_type.clone()).or_insert(0) += 1;
            }
        }
        
        KnowledgeStats {
            total_nodes,
            total_connections,
            avg_activation,
            relation_counts,
        }
    }
    
    /// Clear all knowledge
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.concept_map.clear();
        self.relation_index.clear();
        self.activation_history.clear();
    }
}

/// Knowledge graph statistics
#[derive(Debug, Clone)]
pub struct KnowledgeStats {
    pub total_nodes: usize,
    pub total_connections: usize,
    pub avg_activation: f32,
    pub relation_counts: HashMap<RelationType, usize>,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}