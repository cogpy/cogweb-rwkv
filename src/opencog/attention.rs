//! Attention Allocation System for OpenCog
//! 
//! Implements economic attention allocation mechanisms that direct
//! cognitive resources towards the most important information.

use crate::opencog::atomspace::{AtomSpace, AtomId, Atom};
use std::collections::HashMap;

/// Economic attention value (ECAN) parameters
#[derive(Debug, Clone)]
pub struct AttentionValue {
    /// Short-term importance (STI)
    pub sti: i32,
    /// Long-term importance (LTI) 
    pub lti: i32,
    /// Very-long-term importance (VLTI)
    pub vlti: i32,
}

impl AttentionValue {
    pub fn new(sti: i32, lti: i32, vlti: i32) -> Self {
        Self { sti, lti, vlti }
    }
    
    /// Calculate total importance
    pub fn importance(&self) -> f32 {
        (self.sti as f32 * 0.5 + self.lti as f32 * 0.3 + self.vlti as f32 * 0.2) / 100.0
    }
}

impl Default for AttentionValue {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

/// Attention allocation parameters
#[derive(Debug, Clone)]
pub struct AttentionParams {
    /// Total available attention funds
    pub total_funds: i32,
    /// Minimum STI threshold for attentional focus
    pub sti_threshold: i32,
    /// Rent charged per cycle for atoms in focus
    pub rent_rate: f32,
    /// Wages paid for useful cognitive work
    pub wage_rate: f32,
    /// Decay rate for unused attention
    pub decay_rate: f32,
}

impl Default for AttentionParams {
    fn default() -> Self {
        Self {
            total_funds: 1000000,
            sti_threshold: 50,
            rent_rate: 0.01,
            wage_rate: 0.1,
            decay_rate: 0.99,
        }
    }
}

/// Attention allocation system implementing ECAN
#[derive(Debug)]
pub struct AttentionSystem {
    /// Attention values for each atom
    attention_values: HashMap<AtomId, AttentionValue>,
    /// System parameters
    params: AttentionParams,
    /// Current cycle number
    cycle: u64,
    /// Atoms currently in attentional focus
    attentional_focus: Vec<AtomId>,
}

impl AttentionSystem {
    pub fn new() -> Self {
        Self {
            attention_values: HashMap::new(),
            params: AttentionParams::default(),
            cycle: 0,
            attentional_focus: Vec::new(),
        }
    }
    
    /// Allocate attention based on input tokens
    pub fn allocate(&mut self, input: &[u16]) -> Vec<f32> {
        self.cycle += 1;
        
        // Calculate attention weights based on token frequency, position, novelty
        let mut weights = Vec::with_capacity(input.len());
        
        for (pos, &token) in input.iter().enumerate() {
            let position_weight = 1.0 - (pos as f32 / input.len() as f32) * 0.5;
            let frequency_weight = self.calculate_frequency_weight(token);
            let novelty_weight = self.calculate_novelty_weight(token);
            
            let total_weight = position_weight * frequency_weight * novelty_weight;
            weights.push(total_weight);
        }
        
        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }
        
        weights
    }
    
    /// Update attention values for atoms based on usage
    pub fn update_attention(&mut self, atom_id: AtomId, stimulus: f32) {
        let av = self.attention_values.entry(atom_id).or_default();
        
        // Increase STI based on stimulus
        av.sti += (stimulus * 100.0) as i32;
        
        // Apply economic dynamics
        self.apply_rent(atom_id);
        self.apply_wages(atom_id, stimulus);
    }
    
    /// Calculate frequency-based attention weight
    fn calculate_frequency_weight(&self, token: u16) -> f32 {
        // Simple inverse frequency - rare tokens get more attention
        // In a real system, this would use proper token frequency statistics
        let base_freq = 1.0 / (token as f32 + 1.0);
        (base_freq * 1000.0).min(2.0).max(0.1)
    }
    
    /// Calculate novelty-based attention weight
    fn calculate_novelty_weight(&self, token: u16) -> f32 {
        // Novelty based on how recently we've seen this token
        // This is a simplified version - real implementation would track history
        let novelty = if self.cycle % (token as u64 + 1) == 0 { 0.5 } else { 1.0 };
        novelty
    }
    
    /// Apply rent to atoms in attentional focus
    fn apply_rent(&mut self, atom_id: AtomId) {
        if let Some(av) = self.attention_values.get_mut(&atom_id) {
            if av.sti > self.params.sti_threshold {
                av.sti -= (av.sti as f32 * self.params.rent_rate) as i32;
            }
        }
    }
    
    /// Apply wages for cognitive work
    fn apply_wages(&mut self, atom_id: AtomId, work_value: f32) {
        if let Some(av) = self.attention_values.get_mut(&atom_id) {
            av.sti += (work_value * self.params.wage_rate * 100.0) as i32;
        }
    }
    
    /// Update attentional focus based on STI values
    pub fn update_focus(&mut self, atomspace: &AtomSpace) {
        self.attentional_focus.clear();
        
        // Collect atoms with STI above threshold
        for (&atom_id, av) in &self.attention_values {
            if av.sti > self.params.sti_threshold {
                self.attentional_focus.push(atom_id);
            }
        }
        
        // Sort by STI value (descending)
        self.attentional_focus.sort_by(|&a, &b| {
            let sti_a = self.attention_values.get(&a).map(|av| av.sti).unwrap_or(0);
            let sti_b = self.attention_values.get(&b).map(|av| av.sti).unwrap_or(0);
            sti_b.cmp(&sti_a)
        });
        
        // Limit focus size to prevent cognitive overload
        let max_focus_size = 100;
        if self.attentional_focus.len() > max_focus_size {
            self.attentional_focus.truncate(max_focus_size);
        }
    }
    
    /// Get current attentional focus
    pub fn get_focus(&self) -> &[AtomId] {
        &self.attentional_focus
    }
    
    /// Get attention value for an atom
    pub fn get_attention_value(&self, atom_id: AtomId) -> Option<&AttentionValue> {
        self.attention_values.get(&atom_id)
    }
    
    /// Decay attention values over time
    pub fn decay_attention(&mut self) {
        for av in self.attention_values.values_mut() {
            av.sti = (av.sti as f32 * self.params.decay_rate) as i32;
            av.lti = (av.lti as f32 * self.params.decay_rate.sqrt()) as i32;
        }
    }
    
    /// Spread attention activation between related atoms
    pub fn spread_activation(&mut self, atomspace: &AtomSpace, source_id: AtomId, strength: f32) {
        let related_atoms = atomspace.get_incoming(source_id);
        let spread_amount = (strength * 0.1) as i32;
        
        for atom in related_atoms {
            let av = self.attention_values.entry(atom.id()).or_default();
            av.sti += spread_amount;
        }
    }
    
    /// Get total attention funds in circulation
    pub fn total_attention_funds(&self) -> i32 {
        self.attention_values.values().map(|av| av.sti + av.lti + av.vlti).sum()
    }
    
    /// Normalize attention economy to maintain total funds
    pub fn normalize_economy(&mut self) {
        let current_total = self.total_attention_funds();
        if current_total > 0 && current_total != self.params.total_funds {
            let ratio = self.params.total_funds as f32 / current_total as f32;
            
            for av in self.attention_values.values_mut() {
                av.sti = (av.sti as f32 * ratio) as i32;
                av.lti = (av.lti as f32 * ratio) as i32;
                av.vlti = (av.vlti as f32 * ratio) as i32;
            }
        }
    }
    
    /// Set attention parameters
    pub fn set_params(&mut self, params: AttentionParams) {
        self.params = params;
    }
    
    /// Get current cycle number
    pub fn cycle(&self) -> u64 {
        self.cycle
    }
}

impl Default for AttentionSystem {
    fn default() -> Self {
        Self::new()
    }
}