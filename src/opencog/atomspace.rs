//! AtomSpace implementation for OpenCog integration
//! 
//! The AtomSpace is the central knowledge representation structure in OpenCog.
//! It stores atoms (nodes and links) representing knowledge and relationships.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Unique identifier for atoms
pub type AtomId = u64;

/// Truth value representing certainty and strength of beliefs
#[derive(Debug, Clone, PartialEq)]
pub struct TruthValue {
    /// Strength of belief (0.0 to 1.0)
    pub strength: f32,
    /// Confidence in the strength (0.0 to 1.0)
    pub confidence: f32,
}

impl TruthValue {
    pub fn new(strength: f32, confidence: f32) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
    
    pub fn certain(strength: f32) -> Self {
        Self::new(strength, 1.0)
    }
    
    pub fn uncertain() -> Self {
        Self::new(0.5, 0.0)
    }
}

impl Default for TruthValue {
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

/// Types of atoms in the AtomSpace
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AtomType {
    // Basic node types
    ConceptNode,
    PredicateNode,
    NumberNode,
    WordNode,
    
    // Link types for relationships
    InheritanceLink,
    SimilarityLink,
    ImplicationLink,
    EvaluationLink,
    ListLink,
    
    // RWKV-specific atom types
    TokenNode,
    EmbeddingNode,
    AttentionLink,
    SequenceLink,
}

/// Atom in the AtomSpace - can be a Node or Link
#[derive(Debug, Clone)]
pub enum Atom {
    Node {
        id: AtomId,
        atom_type: AtomType,
        name: String,
        truth_value: TruthValue,
        attention_value: f32,
    },
    Link {
        id: AtomId,
        atom_type: AtomType,
        outgoing: Vec<AtomId>,
        truth_value: TruthValue,
        attention_value: f32,
    },
}

impl Atom {
    pub fn id(&self) -> AtomId {
        match self {
            Atom::Node { id, .. } => *id,
            Atom::Link { id, .. } => *id,
        }
    }
    
    pub fn atom_type(&self) -> &AtomType {
        match self {
            Atom::Node { atom_type, .. } => atom_type,
            Atom::Link { atom_type, .. } => atom_type,
        }
    }
    
    pub fn truth_value(&self) -> &TruthValue {
        match self {
            Atom::Node { truth_value, .. } => truth_value,
            Atom::Link { truth_value, .. } => truth_value,
        }
    }
    
    pub fn attention_value(&self) -> f32 {
        match self {
            Atom::Node { attention_value, .. } => *attention_value,
            Atom::Link { attention_value, .. } => *attention_value,
        }
    }
    
    pub fn set_attention(&mut self, value: f32) {
        match self {
            Atom::Node { attention_value, .. } => *attention_value = value,
            Atom::Link { attention_value, .. } => *attention_value = value,
        }
    }
}

/// The AtomSpace - central knowledge store
#[derive(Debug)]
pub struct AtomSpace {
    atoms: HashMap<AtomId, Atom>,
    next_id: AtomId,
    // Indexes for efficient lookups
    by_type: HashMap<AtomType, HashSet<AtomId>>,
    by_name: HashMap<String, HashSet<AtomId>>,
    incoming_sets: HashMap<AtomId, HashSet<AtomId>>,
}

impl AtomSpace {
    pub fn new() -> Self {
        Self {
            atoms: HashMap::new(),
            next_id: 1,
            by_type: HashMap::new(),
            by_name: HashMap::new(),
            incoming_sets: HashMap::new(),
        }
    }
    
    /// Add a node to the AtomSpace
    pub fn add_node(&mut self, atom_type: AtomType, name: String) -> AtomId {
        let id = self.next_id;
        self.next_id += 1;
        
        let atom = Atom::Node {
            id,
            atom_type: atom_type.clone(),
            name: name.clone(),
            truth_value: TruthValue::default(),
            attention_value: 0.0,
        };
        
        self.atoms.insert(id, atom);
        self.by_type.entry(atom_type).or_default().insert(id);
        self.by_name.entry(name).or_default().insert(id);
        
        id
    }
    
    /// Add a link to the AtomSpace
    pub fn add_link(&mut self, atom_type: AtomType, outgoing: Vec<AtomId>) -> AtomId {
        let id = self.next_id;
        self.next_id += 1;
        
        let atom = Atom::Link {
            id,
            atom_type: atom_type.clone(),
            outgoing: outgoing.clone(),
            truth_value: TruthValue::default(),
            attention_value: 0.0,
        };
        
        self.atoms.insert(id, atom);
        self.by_type.entry(atom_type).or_default().insert(id);
        
        // Update incoming sets
        for &target_id in &outgoing {
            self.incoming_sets.entry(target_id).or_default().insert(id);
        }
        
        id
    }
    
    /// Get an atom by ID
    pub fn get_atom(&self, id: AtomId) -> Option<&Atom> {
        self.atoms.get(&id)
    }
    
    /// Get all atoms of a specific type
    pub fn get_atoms_by_type(&self, atom_type: &AtomType) -> Vec<&Atom> {
        self.by_type
            .get(atom_type)
            .map(|ids| ids.iter().filter_map(|id| self.atoms.get(id)).collect())
            .unwrap_or_default()
    }
    
    /// Find node by name and type
    pub fn find_node(&self, atom_type: &AtomType, name: &str) -> Option<AtomId> {
        self.by_name.get(name)?
            .iter()
            .find(|&&id| {
                if let Some(atom) = self.atoms.get(&id) {
                    atom.atom_type() == atom_type
                } else {
                    false
                }
            })
            .copied()
    }
    
    /// Get incoming links for an atom
    pub fn get_incoming(&self, id: AtomId) -> Vec<&Atom> {
        self.incoming_sets
            .get(&id)
            .map(|ids| ids.iter().filter_map(|id| self.atoms.get(id)).collect())
            .unwrap_or_default()
    }
    
    /// Update truth value of an atom
    pub fn set_truth_value(&mut self, id: AtomId, truth_value: TruthValue) -> bool {
        if let Some(atom) = self.atoms.get_mut(&id) {
            match atom {
                Atom::Node { truth_value: tv, .. } => *tv = truth_value,
                Atom::Link { truth_value: tv, .. } => *tv = truth_value,
            }
            true
        } else {
            false
        }
    }
    
    /// Update attention value of an atom
    pub fn set_attention_value(&mut self, id: AtomId, attention: f32) -> bool {
        if let Some(atom) = self.atoms.get_mut(&id) {
            atom.set_attention(attention);
            true
        } else {
            false
        }
    }
    
    /// Get atoms with highest attention values
    pub fn get_attentional_focus(&self, limit: usize) -> Vec<&Atom> {
        let mut atoms: Vec<&Atom> = self.atoms.values().collect();
        atoms.sort_by(|a, b| b.attention_value().partial_cmp(&a.attention_value()).unwrap_or(std::cmp::Ordering::Equal));
        atoms.into_iter().take(limit).collect()
    }
    
    /// Count total atoms in the AtomSpace
    pub fn size(&self) -> usize {
        self.atoms.len()
    }
    
    /// Clear all atoms
    pub fn clear(&mut self) {
        self.atoms.clear();
        self.by_type.clear();
        self.by_name.clear();
        self.incoming_sets.clear();
        self.next_id = 1;
    }
}

impl Default for AtomSpace {
    fn default() -> Self {
        Self::new()
    }
}